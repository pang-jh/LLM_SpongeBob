import json
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        预训练数据集初始化
        
        参数:
          data_path: 数据文件路径，每行是一个JSON格式的样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回一个样本列表，每个样本为字典格式
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())  # line.strip() 会去掉读取到的行首尾的空白字符（包括空格、换行符等）
                samples.append(data)
        return samples
    # samples 列表中的每个元素都是从文件 f 中读取的一行数据解析而来的 JSON 对象

    def __len__(self):
        # 返回样本总数
        return len(self.samples)
    
    def __getitem__(self, index):
        # 根据索引获取单个样本数据
        sample = self.samples[index]
        # 构建输入文本，加上起始符和结束符
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        # 利用 tokenizer 将文本编码为 token IDs，固定最大长度，进行 padding 和截断
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,  # 如果文本超过最大长度，则截断
            return_tensors='pt'  # 返回的类型为张量
        )
        # 获取输入 token IDs，去除多余的维度
        input_ids = encoding.input_ids.squeeze()
        # 构造损失掩码，标记非填充位置
        # 这段代码通过比较 input_ids 中的每个 token ID 是否等于填充 token ID，生成一个布尔数组（或张量）
        # 有效的 token（非填充位置）对应的值为 True，反之False
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        微调数据集初始化
        
        参数:
          jsonl_path: 数据文件路径，每行是一个 JSON 格式的对话样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回样本列表
        self.samples = self.load_data(jsonl_path)
        # 定义开始和结束标记的 token ID（不添加特殊 token）
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        # 返回样本总数
        return len(self.samples)
    
    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self, conversations):
        """
        构建符合 ChatML 格式的对话提示
        
        参数:
          conversations: 对话轮次列表，每个元素为包含 'content' 的字典
        
        返回:
          prompt: 拼接后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 偶数轮为用户，奇数轮为助手
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 使用分词器提供的模板方法构建对话提示
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def _generate_loss_mask(self, input_ids):
        """
        根据输入 token IDs 生成损失掩码。
        只有位于 <s>assistant\n 与 </s>\n 之间的 token 被标记为 1，其余为 0
        
        参数:
          input_ids: 一个整数列表，表示输入 token IDs
        
        返回:
          loss_mask: 一个与 input_ids 长度相同的列表，1 表示计算损失的位置，0 表示忽略
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        # 遍历整个输入序列
        while i < len(input_ids):
            # 检查当前位置是否匹配开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 记录开始位置，排除 bos 部分
                start = i + len(self.bos_id)
                end = start
                # 从开始位置向后查找结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将开始标记之后到结束标记位置之间的 token 标记为 1（参与损失计算）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 更新索引：跳过整个对话部分（包括结束标记）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index):
        # 获取对应索引的样本
        sample = self.samples[index]
        # 利用对话轮次构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 对对话提示进行编码，并限制最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 若不足最大长度则补齐 pad_token_id
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 根据输入 token IDs 生成动态的损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据：X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask
