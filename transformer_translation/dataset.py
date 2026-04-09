"""
数据集处理模块
负责将原始句对数据转换为模型可用的张量格式
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional

from vocabulary import (
    Vocabulary, Tokenizer,
    PAD_IDX, BOS_IDX, EOS_IDX
)


class TranslationDataset(Dataset):
    """机器翻译数据集"""
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_lang: str = 'en',
        tgt_lang: str = 'zh',
        max_src_len: int = 100,
        max_tgt_len: int = 100,
    ):
        """
        Args:
            pairs: [(src_text, tgt_text), ...] 原始句对
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            src_lang: 源语言代码 ('en' 或 'zh')
            tgt_lang: 目标语言代码 ('en' 或 'zh')
            max_src_len: 源序列最大长度
            max_tgt_len: 目标序列最大长度
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = Tokenizer()
        
        self.data = self._process_pairs(pairs)
        print(f"数据集大小: {len(self.data)} 个句对")
    
    def _tokenize(self, text: str, lang: str) -> List[str]:
        """根据语言选择分词方式"""
        if lang == 'zh':
            return self.tokenizer.tokenize_zh(text)
        else:
            return self.tokenizer.tokenize_en(text)
    
    def _process_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """处理所有句对，转为索引张量"""
        processed = []
        skipped = 0
        
        for src_text, tgt_text in pairs:
            src_tokens = self._tokenize(src_text, self.src_lang)
            tgt_tokens = self._tokenize(tgt_text, self.tgt_lang)
            
            # 过滤过长的句子
            if len(src_tokens) > self.max_src_len or len(tgt_tokens) > self.max_tgt_len:
                skipped += 1
                continue
            
            # 编码为索引
            src_indices = self.src_vocab.encode(src_tokens)
            tgt_indices = self.tgt_vocab.encode(tgt_tokens)
            
            # 转为张量
            src_tensor = torch.tensor(src_indices, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
            
            processed.append((src_tensor, tgt_tensor))
        
        if skipped > 0:
            print(f"跳过 {skipped} 个过长的句对")
        
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批次整理函数：对序列进行填充
    
    Returns:
        src_padded: (batch, src_len) 填充后的源序列
        tgt_input: (batch, tgt_len) 解码器输入（去掉最后一个token）
        tgt_output: (batch, tgt_len) 解码器期望输出（去掉第一个token）
        src_lengths: 源序列实际长度
    """
    src_batch, tgt_batch = zip(*batch)
    
    # 添加 BOS 和 EOS 标记
    src_batch_with_special = [
        torch.cat([torch.tensor([BOS_IDX]), src, torch.tensor([EOS_IDX])])
        for src in src_batch
    ]
    tgt_batch_with_special = [
        torch.cat([torch.tensor([BOS_IDX]), tgt, torch.tensor([EOS_IDX])])
        for tgt in tgt_batch
    ]
    
    # 填充到相同长度
    src_padded = pad_sequence(src_batch_with_special, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch_with_special, batch_first=True, padding_value=PAD_IDX)
    
    # 解码器输入：去掉最后一个 token（EOS）
    # 解码器输出（标签）：去掉第一个 token（BOS）
    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]
    
    src_lengths = torch.tensor([s.size(0) for s in src_batch_with_special])
    
    return src_padded, tgt_input, tgt_output, src_lengths


def create_dataloaders(
    train_pairs: List[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 32,
    src_lang: str = 'en',
    tgt_lang: str = 'zh',
    max_src_len: int = 100,
    max_tgt_len: int = 100,
    val_pairs: Optional[List[Tuple[str, str]]] = None,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建训练和验证数据加载器
    
    Args:
        train_pairs: 训练句对
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        batch_size: 批大小
        src_lang: 源语言
        tgt_lang: 目标语言
        max_src_len: 源序列最大长度
        max_tgt_len: 目标序列最大长度
        val_pairs: 验证句对（如果为None则从训练集中划分）
        val_ratio: 从训练集划分验证集的比例
        num_workers: 数据加载线程数
    
    Returns:
        (train_loader, val_loader)
    """
    # 如果没有提供验证集，从训练集中拆分
    if val_pairs is None and val_ratio > 0:
        import random
        random.shuffle(train_pairs)
        split_idx = int(len(train_pairs) * (1 - val_ratio))
        val_pairs = train_pairs[split_idx:]
        train_pairs = train_pairs[:split_idx]
    
    print(f"\n训练集大小: {len(train_pairs)}")
    if val_pairs:
        print(f"验证集大小: {len(val_pairs)}")
    
    # 创建训练数据集
    train_dataset = TranslationDataset(
        train_pairs, src_vocab, tgt_vocab,
        src_lang, tgt_lang, max_src_len, max_tgt_len
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    # 创建验证数据集
    val_loader = None
    if val_pairs:
        val_dataset = TranslationDataset(
            val_pairs, src_vocab, tgt_vocab,
            src_lang, tgt_lang, max_src_len, max_tgt_len
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    from vocabulary import SAMPLE_EN_ZH_PAIRS, build_vocab_from_pairs
    
    # 构建词汇表
    src_vocab, tgt_vocab = build_vocab_from_pairs(SAMPLE_EN_ZH_PAIRS)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        SAMPLE_EN_ZH_PAIRS,
        src_vocab,
        tgt_vocab,
        batch_size=4,
        val_ratio=0.1
    )
    
    # 查看一个批次
    for src, tgt_in, tgt_out, src_lens in train_loader:
        print(f"\n批次形状:")
        print(f"  src:     {src.shape}")
        print(f"  tgt_in:  {tgt_in.shape}")
        print(f"  tgt_out: {tgt_out.shape}")
        print(f"  src_len: {src_lens}")
        break
