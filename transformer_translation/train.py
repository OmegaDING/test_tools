"""
训练脚本
支持：
  - 学习率预热 + 余弦退火调度
  - 标签平滑交叉熵
  - 梯度裁剪
  - 验证集评估
  - 最优模型保存 & 早停
  - 训练日志记录
"""

import os
import sys
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_model import Transformer, count_parameters
from vocabulary import (
    Vocabulary, Tokenizer, build_vocab_from_pairs,
    PAD_IDX, BOS_IDX, EOS_IDX,
    SAMPLE_EN_ZH_PAIRS
)
from dataset import create_dataloaders
from config import TranslationConfig, get_small_config


# ============================================================
# 学习率调度器
# ============================================================

def get_warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """预热 + 余弦退火学习率调度"""
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_noam_schedule(optimizer, d_model: int, warmup_steps: int):
    """原论文中的 Noam 调度器"""
    def lr_lambda(step: int):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 标签平滑损失
# ============================================================

class LabelSmoothingLoss(nn.Module):
    """带标签平滑的交叉熵损失"""
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            targets: (batch * seq_len,)
        """
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # 忽略 padding 位置
        non_pad_mask = targets != self.padding_idx
        
        # 对数概率
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 平滑标签
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_targets.scatter_(1, targets.unsqueeze(1).clamp(0), self.confidence)
            smooth_targets[:, self.padding_idx] = 0
            smooth_targets[~non_pad_mask] = 0
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = loss[non_pad_mask].mean()
        return loss


# ============================================================
# 训练和验证
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None,
                grad_clip: float = 1.0, log_every: int = 50):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, (src, tgt_in, tgt_out, _) in enumerate(loader):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        
        # 前向传播
        logits = model(src, tgt_in, pad_idx=PAD_IDX)  # (batch, tgt_len, vocab)
        
        # 计算损失
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        num_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / max(total_tokens, 1)
            print(f"  Batch [{batch_idx+1:>4}/{len(loader)}]  "
                  f"Loss: {avg_loss:.4f}  "
                  f"PPL: {math.exp(min(avg_loss, 20)):.2f}  "
                  f"LR: {current_lr:.2e}  "
                  f"Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for src, tgt_in, tgt_out, _ in loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        
        logits = model(src, tgt_in, pad_idx=PAD_IDX)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
        
        num_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ============================================================
# 检查点保存与加载
# ============================================================

def save_checkpoint(model, optimizer, epoch, val_loss, config, path):
    """保存训练检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config.__dict__,
    }, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))


# ============================================================
# 主训练函数
# ============================================================

def train(config: TranslationConfig = None, data_pairs=None):
    """
    主训练入口
    
    Args:
        config: 训练配置，默认使用 get_small_config()
        data_pairs: 训练数据句对列表，默认使用内置示例数据
    """
    if config is None:
        config = get_small_config()
    
    if data_pairs is None:
        data_pairs = SAMPLE_EN_ZH_PAIRS
        print(f"使用内置示例数据集（{len(data_pairs)} 个句对）")
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    config.display()
    
    # ── 构建词汇表 ──────────────────────────────────────────
    print("\n[1/4] 构建词汇表...")
    src_vocab, tgt_vocab = build_vocab_from_pairs(
        data_pairs,
        src_min_freq=config.src_min_freq,
        tgt_min_freq=config.tgt_min_freq,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang
    )
    src_vocab.save(config.src_vocab_path)
    tgt_vocab.save(config.tgt_vocab_path)
    
    # ── 准备数据 ──────────────────────────────────────────
    print("\n[2/4] 准备数据集...")
    train_loader, val_loader = create_dataloaders(
        list(data_pairs),  # 复制一份，避免被 shuffle 修改原数据
        src_vocab,
        tgt_vocab,
        batch_size=config.batch_size,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
        val_ratio=config.val_ratio,
    )
    
    # ── 构建模型 ──────────────────────────────────────────
    print("\n[3/4] 构建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        max_len=config.max_len,
        dropout=config.dropout
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=PAD_IDX,
        smoothing=config.label_smoothing
    )
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = None
    if config.warmup_steps > 0:
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_warmup_cosine_schedule(
            optimizer, config.warmup_steps, total_steps
        )
    
    # ── 开始训练 ──────────────────────────────────────────
    print("\n[4/4] 开始训练...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 日志文件
    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    log_f = open(config.log_file, 'w', encoding='utf-8')
    log_f.write("epoch,train_loss,val_loss,train_ppl,val_ppl\n")
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch}/{config.num_epochs}]")
        print(f"{'='*60}")
        
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler,
            grad_clip=config.grad_clip,
            log_every=config.log_every
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss = float('inf')
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20)) if val_loss != float('inf') else float('inf')
        
        print(f"\n  训练损失: {train_loss:.4f}  PPL: {train_ppl:.2f}")
        if val_loader is not None:
            print(f"  验证损失: {val_loss:.4f}  PPL: {val_ppl:.2f}")
        print(f"  用时: {epoch_time:.1f}s")
        
        # 写入日志
        log_f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_ppl:.4f},{val_ppl:.4f}\n")
        log_f.flush()
        
        # 保存最优模型
        monitor_loss = val_loss if val_loader is not None else train_loss
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, monitor_loss, config, config.best_model_path)
            print(f"  ✓ 保存最优模型 (损失: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, monitor_loss, config, config.last_model_path)
        
        # 早停
        if config.early_stopping_patience > 0 and patience_counter >= config.early_stopping_patience:
            print(f"\n早停触发（已连续 {patience_counter} 个 epoch 无改善）")
            break
    
    log_f.close()
    
    # 保存最终模型
    save_checkpoint(model, optimizer, epoch, train_loss, config, config.last_model_path)
    config.save()
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最优验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {config.best_model_path}")
    print(f"训练日志: {config.log_file}")
    print(f"{'='*60}")
    
    return model, src_vocab, tgt_vocab


# ============================================================
# 命令行接口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer 翻译模型训练")
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（JSON）')
    parser.add_argument('--size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='预设模型大小')
    parser.add_argument('--data', type=str, default=None,
                        help='训练数据文件路径（每行格式：源语句\\t目标语句）')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='源语言 (en/zh)')
    parser.add_argument('--tgt_lang', type=str, default='zh',
                        help='目标语言 (en/zh)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='输出目录')
    return parser.parse_args()


def load_data_from_file(filepath: str, delimiter: str = '\t') -> list:
    """从文件加载句对数据"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(delimiter)
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                print(f"警告: 第 {line_no} 行格式不正确: {line[:50]}")
    print(f"从文件加载 {len(pairs)} 个句对: {filepath}")
    return pairs


if __name__ == "__main__":
    args = parse_args()
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        config = TranslationConfig.load(args.config)
        print(f"从文件加载配置: {args.config}")
    else:
        from config import get_small_config, get_medium_config, get_large_config
        if args.size == 'small':
            config = get_small_config()
        elif args.size == 'medium':
            config = get_medium_config()
        else:
            config = get_large_config()
    
    # 命令行参数覆盖配置
    config.src_lang = args.src_lang
    config.tgt_lang = args.tgt_lang
    config.output_dir = args.output_dir
    config.src_vocab_path = os.path.join(args.output_dir, 'src_vocab.json')
    config.tgt_vocab_path = os.path.join(args.output_dir, 'tgt_vocab.json')
    config.best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    config.last_model_path = os.path.join(args.output_dir, 'last_model.pt')
    config.log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # 加载数据
    data_pairs = None
    if args.data and os.path.exists(args.data):
        data_pairs = load_data_from_file(args.data)
    else:
        print("未指定数据文件，使用内置示例数据集")
    
    # 开始训练
    train(config, data_pairs)
