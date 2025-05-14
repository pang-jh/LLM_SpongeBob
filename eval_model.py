import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import SpongeBob
from Config import LLMConfig


def init_model(args):
     tokenizer = AutoTokenizer.from_pretrained('./spongebob_tokenizer')
     if args.model_mode == 1:
          ckp = f'./{args.save_dir}/SFT.pth'
     if args.model_mode == 0:
          ckp = f'./{args.save_dir}/pretrain.pth'

     model = SpongeBob(LLMConfig(
          max_seq_len=args.max_seq_len,
     ))

     state_dict = torch.load(ckp, map_location=args.device)
     model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

     print(f'模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
     return model.eval().to(args.device), tokenizer




def main():
     parser=argparse.ArgumentParser()
     parser.add_argument('--save_dir', default='results', type=str)
     parser.add_argument('--temperature', default=0.85, type=float)
     parser.add_argument('--top_p', default=0.85, type=float)
     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
     parser.add_argument('--max_seq_len', default=8192, type=int)
     parser.add_argument('--history_cnt', default=0, type=int)
     parser.add_argument('--stream', default=True, type=bool)
     parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型")
     
     args = parser.parse_args()
     model, tokenizer = init_model(args)
     messages=[]

     while True:
          # 获取用户输入
          prompt = input('👶: ')  # 手动输入对话内容

          messages = messages[-args.history_cnt:] if args.history_cnt else []
          messages.append({"role": "user", "content": prompt})

          new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
          )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)
          print('new_prompt:', new_prompt)
          
          with torch.no_grad():
               x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
               outputs = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_seq_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=True,
                    pad_token_id=tokenizer.pad_token_id,
                    rp=1.3
               )

               print('🤖️: ', end='')  # 打印机器人图标，不换行
               try:
                    history_idx = 0  # 初始化历史索引，用于跟踪已打印的答案长度
                    for y in outputs:  # 遍历模型生成的每个输出
                         # y[0]为input_ids
                         answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)  # 解码输出，跳过特殊标记
                         if (answer and answer[-1] == '�') or not answer:  # 检查答案是否有效
                              continue  # 如果答案无效，则跳过当前循环
                         print(answer[history_idx:], end='', flush=True)  # 打印有效答案的剩余部分，不换行
                         history_idx = len(answer)  # 更新历史索引为当前答案的长度
               except StopIteration:  # 捕获停止迭代异常
                    print("No answer")  # 如果没有答案，打印提示信息
               print('\n')

          messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()