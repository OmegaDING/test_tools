"""
训练配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranslationConfig:
    """翻译模型训练配置"""
    
    # ── 语言设置 ──────────────────────────────────────────
    src_lang: str = 'en'          # 源语言 ('en' 或 'zh')
    tgt_lang: str = 'zh'          # 目标语言 ('en' 或 'zh')
    
    # ── 模型超参数 ────────────────────────────────────────
    d_model: int = 256            # 模型维度
    num_heads: int = 8            # 注意力头数
    d_ff: int = 512               # 前馈网络维度
    num_encoder_layers: int = 3   # 编码器层数
    num_decoder_layers: int = 3   # 解码器层数
    max_len: int = 128            # 最大序列长度
    dropout: float = 0.1          # Dropout率
    
    # ── 训练超参数 ────────────────────────────────────────
    batch_size: int = 16          # 批大小
    num_epochs: int = 50          # 训练轮数
    learning_rate: float = 1e-3  # 初始学习率
    warmup_steps: int = 400       # 学习率预热步数（0表示不使用）
    weight_decay: float = 1e-4   # 权重衰减
    grad_clip: float = 1.0        # 梯度裁剪阈值
    label_smoothing: float = 0.1  # 标签平滑系数
    
    # ── 数据设置 ──────────────────────────────────────────
    max_src_len: int = 100        # 源序列最大长度（过滤用）
    max_tgt_len: int = 100        # 目标序列最大长度（过滤用）
    val_ratio: float = 0.1        # 验证集比例
    src_min_freq: int = 1         # 源语言最小词频
    tgt_min_freq: int = 1         # 目标语言最小词频
    
    # ── 路径设置 ──────────────────────────────────────────
    output_dir: str = 'checkpoints'       # 输出目录
    src_vocab_path: str = 'checkpoints/src_vocab.json'
    tgt_vocab_path: str = 'checkpoints/tgt_vocab.json'
    best_model_path: str = 'checkpoints/best_model.pt'
    last_model_path: str = 'checkpoints/last_model.pt'
    log_file: str = 'checkpoints/training_log.txt'
    
    # ── 训练控制 ──────────────────────────────────────────
    save_every: int = 5           # 每N个epoch保存一次
    log_every: int = 10           # 每N个batch打印一次日志
    early_stopping_patience: int = 10   # 早停耐心值（0表示不使用）
    seed: int = 42                # 随机种子
    
    # ── 推理设置 ──────────────────────────────────────────
    max_translate_len: int = 100  # 翻译时最大生成长度
    beam_size: int = 4            # Beam Search宽度（1表示贪婪解码）
    length_penalty: float = 0.6   # 长度惩罚系数（Beam Search时使用）
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
    
    def display(self):
        """打印配置信息"""
        print("=" * 50)
        print("训练配置")
        print("=" * 50)
        for key, val in self.__dict__.items():
            print(f"  {key:<28} = {val}")
        print("=" * 50)
    
    def save(self, path: Optional[str] = None):
        """保存配置到文件"""
        import json
        save_path = path or os.path.join(self.output_dir, 'config.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
        print(f"配置已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'TranslationConfig':
        """从文件加载配置"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = cls()
        for key, val in data.items():
            if hasattr(config, key):
                setattr(config, key, val)
        return config


# 预设配置
def get_small_config() -> TranslationConfig:
    """小型配置（适合示例数据快速训练）"""
    config = TranslationConfig()
    config.d_model = 128
    config.num_heads = 4
    config.d_ff = 256
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.batch_size = 8
    config.num_epochs = 100
    config.learning_rate = 5e-4
    config.warmup_steps = 100
    return config


def get_medium_config() -> TranslationConfig:
    """中型配置（适合中等规模数据集）"""
    config = TranslationConfig()
    config.d_model = 256
    config.num_heads = 8
    config.d_ff = 512
    config.num_encoder_layers = 3
    config.num_decoder_layers = 3
    config.batch_size = 32
    config.num_epochs = 50
    config.learning_rate = 1e-3
    config.warmup_steps = 400
    return config


def get_large_config() -> TranslationConfig:
    """大型配置（接近标准Transformer）"""
    config = TranslationConfig()
    config.d_model = 512
    config.num_heads = 8
    config.d_ff = 2048
    config.num_encoder_layers = 6
    config.num_decoder_layers = 6
    config.batch_size = 64
    config.num_epochs = 30
    config.learning_rate = 1e-4
    config.warmup_steps = 4000
    config.dropout = 0.1
    return config


if __name__ == "__main__":
    config = get_small_config()
    config.display()
