<img src="SpongeBob.jpg" alt="outline" width="400"/>

# LLM_SpongeBob
本次项目实现了从0到1的手写大模型及训练，如若认真完成本次项目，你将会对大模型整个流程有非常透彻的了解，这种理解程度不是普通的调包项目所能及的，若感觉比较吃力，可以按照所示流程简单跑通，也会有所裨益!

## 基础组件：

1、模型配置代码：Config.py

2、模型架构代码（定义了模型的架构和训练推理全流程细节）：model.py

3、数据集dataset类定义：dataset.py

定义了dataset类，方便之后进行数据加载，有Pretrain和SFT两个类

## 预训练：

### 预训练数据集:
数据集已上传至modelscope、直接当前目录下在命令行运行

```bash
pip install modelscope
modelscope download --dataset Harris/pretrainSpongeBob pretrain.jsonl --local_dir ./
```

### 分词器训练代码：train_tokenizer.py 

需在当前目录下运行

```bash
python train_tokenizer.py 
```

开始运行分词器训练。最终训好的tokenizer，可自取使用(spongebob_tokenizer)

### 预训练代码：pretrain.py

执行训练过程

```bash
python pretrain.py 
```
pretrain后到模型文件，可直接取用(pretrain.pth)(见文末)

### 推理代码：eval_model.py

执行推理过程

```bash
python eval_model.py --model_mode 0
```

## SFT训练：
SFT的代码大量继承了Pretrain的代码，仅仅数据加载做了改变，SFT类数据集定义参考dataset.py文件

### SFT数据集
sft512.jsonl(7.1G)，由匠数科技的SFT数据(24G)清洗而成，筛选出了总长度小于512的部分。

通过modelscope命令下载
```bash
modelscope download --dataset Harris/pretrainSpongeBob sft_512.jsonl --local_dir ./
```

### SFT训练代码：SFT.py

执行训练过程

```bash
python SFT.py 
```
训练后的模型权重，可直接取用(SFT.pth)

### 推理代码：eval_model.py

执行推理过程

```bash
python eval_model.py --model_mode 1
```

## 模型长文本能力训练

继承SFT训练代码，唯一不同是此次使用长度为512-1024的问答对进行训练，让模型在该区间内具备能力

### SFT长文本数据集
sft_1024.jsonl(5.2G)，下载命令：
```bash
modelscope download --dataset Harris/pretrainSpongeBob sft_1024.jsonl --local_dir ./
```
### 修改
相比于SFT.py有几处需要修改

max_seq_len参数需要修改为1024

data_path参数需要修改为sft_1024.jsonl

init_model函数中加载时，应该加载SFT.pth(即上一步的SFT模型)

train_epoch函数中的save部分，建议保存为SFT_long.pth以和SFT做对比

依据本人硬件条件，batch_size要适当改小

### SFT长文本训练代码：SFT.py
执行训练过程

```bash
python SFT.py 
```
训练后的模型权重，可直接取用(SFT_1024.pth)

### 推理代码：eval_model.py

执行推理过程，和SFT相同，将推理加载模型改为训好的SFT_long.pth即可

```bash
python eval_model.py --model_mode 1
```

## R1思维链蒸馏

### R1蒸馏数据集
下载命令：
```bash
modelscope download --dataset Harris/pretrainSpongeBob r1_1024.jsonl --local_dir ./
```
鉴于我们的tokenizer对<think></think>编码效率低，需要4个token，因此模型对学习这种范式会略显困难，为了优先学习这种范式，我们会手动加大这些token的损失惩罚。

### 蒸馏代码：distill.py

其和SFT唯一区别是修改了loss针对思维链token的损失惩罚

执行训练过程

```bash
python distill.py --use_wandb
```
训练后的模型权重，可直接取用

distill.pth——使用SFT.pth作为基座训练

distill_long.pth——使用SFT_1024.pth作为基座训练

### 推理代码：eval_model.py

直接通过修改eval_model.py加载相应模型

```bash
python eval_model.py --model_mode 2
```
# 注：
模型参数文件，链接
```bash
https://pan.baidu.com/s/1qCn3EohZEX9yy4zlsY-wiA?pwd=h1ue
```
提取码: h1ue
