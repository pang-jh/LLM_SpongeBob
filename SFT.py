import os
import argparse
import time
import math
import torch
from torch import optim, nn
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import SpongeBob
from Config import LLMConfig
from dataset import SFTDataset

def get_lr(current_step, total_steps, lr, warmup_iters=0):
     return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
     loss_fct = nn.CrossEntropyLoss(reduction='none')
     start_time=time.time()

     for step, (X, Y, loss_mask) in enumerate(train_loader):
          X = X.to(args.device)
          Y = Y.to(args.device)
          loss_mask = loss_mask.to(args.device)

          lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate, args.warmup_iters)
          for param_group in optimizer.param_groups:
               param_group['lr'] = lr

          with ctx:
               res = model(X)
               loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
               ).view(Y.size())
               loss = (loss * loss_mask).sum() / loss_mask.sum()
               loss = loss / args.accumulation_steps

          scaler.scale(loss).backward()
          if (step + 1) % args.accumulation_steps == 0:
               scaler.unscale_(optimizer)
               torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

               scaler.step(optimizer)
               scaler.update()
               optimizer.zero_grad(set_to_none=True)
          if step % args.log_step == 0:
               spend_time = time.time() - start_time
               print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                         epoch + 1,
                         args.epochs,
                         step,
                         iter_per_epoch,
                         loss.item() * args.accumulation_steps,
                         optimizer.param_groups[-1]['lr'],
                         spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60)
                         )
               if (wandb is not None) :
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

          
          if (step + 1) % args.save_step == 0:
               model.eval()
               ckp = f'{args.save_dir}/SFT.pth'

               if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
               else:
                    state_dict = model.state_dict()

               torch.save(state_dict, ckp)
               model.train()

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('spongebob_tokenizer')
    model = SpongeBob(lm_config)
    ckp = f'./results/pretrain.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer

if __name__ == "__main__":
     parser = argparse.ArgumentParser()

     parser.add_argument("--save_dir", type=str, default="results")
     parser.add_argument("--epochs", type=int, default=1)
     parser.add_argument("--batch_size", type=int, default=84)
     parser.add_argument("--learning_rate", type=float, default=5e-4)
     parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
     parser.add_argument("--use_wandb", type=bool,default=True)
     parser.add_argument("--dtype", type=str, default="bfloat16")
     parser.add_argument("--wandb_project", type=str, default="SpongeBob-SFT")
     parser.add_argument("--num_workers", type=int, default=1)
     parser.add_argument("--accumulation_steps", type=int, default=2)
     parser.add_argument("--grad_clip", type=float, default=1.0)
     parser.add_argument("--warmup_iters", type=int, default=1700)
     parser.add_argument("--log_step", type=int, default=10)
     parser.add_argument("--save_step", type=int, default=1000)
     parser.add_argument('--max_seq_len', default=512, type=int)
     parser.add_argument("--data_path", type=str, default="sft_512.jsonl")
     
     args = parser.parse_args()

     lm_config = LLMConfig(max_seq_len=args.max_seq_len)
     
     args.save_dir = os.path.join(args.save_dir)
     os.makedirs(args.save_dir, exist_ok=True)
     # os.makedirs(args.out_dir, exist_ok=True)
     tokens_per_iter = args.batch_size * lm_config.max_seq_len
     torch.manual_seed(1337)
     device_type = "cuda" if "cuda" in args.device else "cpu"

     args.wandb_run_name = f"SpongeBob-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

     ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

     if args.use_wandb :
          import wandb

          wandb.init(project=args.wandb_project, name=args.wandb_run_name)
     else:
          wandb = None

     model, tokenizer = init_model(lm_config)

     train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
     train_loader = DataLoader(
          train_ds,
          batch_size=args.batch_size,
          pin_memory=True,
          drop_last=False,
          shuffle=False,
          num_workers=args.num_workers,
     )

     scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)


     iter_per_epoch = len(train_loader)
     for epoch in range(args.epochs):
          train_epoch(epoch, wandb)