import random
import json
from tokenizers import (
     decoders,
     models,
     normalizers,
     pre_tokenizers,
     processors,
     trainers,
     Tokenizer,
)

import os

# 定义一个训练分词器的函数
def train_tokenizer():
    # 定义一个从JSONL文件中读取文本的生成器
    def read_texts_from_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            # 一行一行读取文件
            for line in f:
                # 解析每一行的JSON数据，提取文本部分
                data = json.loads(line)
                yield data["text"]

    # 设置数据文件路径
    data_path = "pretrain.jsonl"

    # 创建一个基于BPE（Byte Pair Encoding）模型的分词器
    # 只有models.BPE，没有models.BBPE，当选择了ByteLevel等价于选择了BBPE
    tokenizer = Tokenizer(models.BPE())
    # 设置分词器的预分词器为ByteLevel
    # ByteLevel 将文本视为字节序列，而不是字符序列。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # ByteLevel相当于用utf-8的形式打散

    # 定义特殊符号（包括未知符号、开始符号和结束符号）
    special_tokens = ["<ukn>", "<s>", "</s>"]

    # 创建BPE训练器，设置词表大小和特殊符号
    # BPE词表的合并次数是根据训练数据动态确定的，以生成一个大小为6400的词表。
    trainer = trainers.BpeTrainer(
        vocab_size=6400,  # 词表大小
        special_tokens=special_tokens,  # 特殊符号
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始化字母表，用utf-8编码作为基本单位
    )

    # 从JSONL文件中读取文本
    texts = read_texts_from_jsonl(data_path)

    # 用读取的文本数据训练分词器
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置分词器的解码器为ByteLevel解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 确保特殊符号的token_id设置正确
    assert tokenizer.token_to_id("<ukn>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 设置保存分词器的目录
    tokenizer_dir = "./spongebob_tokenizer"
    # 如果目录不存在，则创建目录
    os.makedirs(tokenizer_dir, exist_ok=True)
    # 保存训练好的分词器到指定目录
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./spongebob_tokenizer")

    # 配置字典，用于存储分词器的相关配置信息
    config = {
        "add_bos_token": False,  # 不添加BOS（Beginning of Sentence）标记
        "add_eos_token": False,  # 不添加EOS（End of Sentence）标记
        "add_prefix_space": False,  # 不添加前缀空格
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",  # 未知符号
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "1": {
                "content": "<s>",  # 句子开始符号
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "2": {
                "content": "</s>",  # 句子结束符号
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
        "addtional_special_tokens": [],  # 额外的特殊符号（为空）
        "bos_token": "<s>",  # 开始符号
        "eos_token": "</s>",  # 结束符号
        "clean_up_tokenization_spaces": False,  # 不清理空格
        "legacy": True,  # 兼容旧版
        "model_max_length": 32768,  # 模型最大长度
        "pad_token": "<unk>",  # 填充符号
        "sp_model_kwargs": {},  # 子词模型的其他参数（为空）
        "spaces_between_special_tokens": False,  # 不在特殊符号之间添加空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 使用的分词器类型
        "unk_token": "<unk>",  # 未知符号的token
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 SpongeBob，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}",
    }

    # 将配置保存到分词器目录下
    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# 主函数调用，启动训练过程
if __name__ == "__main__":
     train_tokenizer()
