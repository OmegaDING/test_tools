# Transformer 机器翻译

基于 PyTorch 从零实现的 Transformer 翻译模型，参考论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

## 功能特性

- ✅ 完整的 Transformer 架构（多头注意力、位置编码、编解码器）
- ✅ 贪婪解码（Greedy Decoding）
- ✅ Beam Search 解码
- ✅ 学习率预热 + 余弦退火调度
- ✅ 标签平滑交叉熵损失
- ✅ 梯度裁剪 & 早停机制
- ✅ 内置英译中示例数据（100+ 句对）
- ✅ 支持自定义数据集
- ✅ 交互式命令行翻译

## 项目结构

```
transformer_translation/
├── transformer_model.py   # Transformer 模型核心实现
├── vocabulary.py          # 词汇表 & 分词工具（含示例数据）
├── dataset.py             # 数据集处理 & DataLoader
├── config.py              # 训练配置（小/中/大型预设）
├── train.py               # 训练脚本
├── translate.py           # 推理翻译脚本
└── README.md              # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
```

> 推荐 PyTorch >= 1.10，支持 CPU 和 CUDA。

### 2. 一键演示（训练 + 翻译）

直接运行 `translate.py`，若无已训练模型会自动训练并展示翻译结果：

```bash
cd transformer_translation
python translate.py
```

### 3. 单独训练

```bash
# 使用内置示例数据（小型模型，快速训练）
python train.py

# 指定预设模型大小
python train.py --size small    # 快速验证
python train.py --size medium   # 中等规模
python train.py --size large    # 接近原论文规模

# 使用自定义数据集（TSV格式：每行 "英文\t中文"）
python train.py --data my_data.txt --src_lang en --tgt_lang zh

# 指定训练轮数和批大小
python train.py --epochs 100 --batch_size 16 --lr 5e-4
```

训练数据文件格式（Tab 分隔）：
```
hello	你好
good morning	早上好
how are you	你好吗
```

### 4. 翻译推理

```bash
# 交互式翻译
python translate.py

# 翻译单句
python translate.py --text "hello world"

# 使用贪婪解码（更快）
python translate.py --method greedy --text "thank you"

# 批量翻译文件
python translate.py --input sentences.txt --output results.txt

# 指定 Beam Search 宽度
python translate.py --beam_size 5 --text "good morning"

# 指定模型路径
python translate.py --model checkpoints/best_model.pt --text "i love you"
```

交互式模式内特殊命令：
- `--greedy` — 切换到贪婪解码
- `--beam`  — 切换到 Beam Search
- `quit` / `exit` — 退出

---

## 模型架构

```
输入句子
    │
[词嵌入 + 位置编码]
    │
┌───▼──────────────────────────────┐
│         编码器 (×N层)             │
│  ┌─────────────────────────────┐ │
│  │  多头自注意力 (Masked)       │ │
│  │  残差连接 + 层归一化         │ │
│  │  位置前馈网络               │ │
│  │  残差连接 + 层归一化         │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
    │
    │  编码器输出
    │
┌───▼──────────────────────────────┐
│         解码器 (×N层)             │
│  ┌─────────────────────────────┐ │
│  │  多头因果自注意力            │ │
│  │  残差连接 + 层归一化         │ │
│  │  编码器-解码器交叉注意力     │ │
│  │  残差连接 + 层归一化         │ │
│  │  位置前馈网络               │ │
│  │  残差连接 + 层归一化         │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
    │
[线性投影 + Softmax]
    │
输出词概率
```

## 配置说明

| 参数 | 小型 | 中型 | 大型（原论文） |
|------|------|------|----------------|
| `d_model` | 128 | 256 | 512 |
| `num_heads` | 4 | 8 | 8 |
| `d_ff` | 256 | 512 | 2048 |
| `num_layers` | 2 | 3 | 6 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| 参数量（约） | ~1M | ~5M | ~65M |

可在 `config.py` 中自定义所有超参数：

```python
from config import TranslationConfig

config = TranslationConfig()
config.d_model = 256
config.num_heads = 8
config.num_epochs = 100
config.batch_size = 32
config.learning_rate = 1e-3
```

## 使用自定义数据集

准备 TSV 格式的平行语料：

```
# data.txt （# 开头为注释，自动忽略）
hello	你好
good morning	早上好
i love machine translation	我爱机器翻译
```

训练：

```bash
python train.py --data data.txt --src_lang en --tgt_lang zh --size medium
```

若使用中译英，交换语言标识即可：

```bash
python train.py --data data.txt --src_lang zh --tgt_lang en
```

## 检查点文件说明

训练后会在 `checkpoints/` 目录生成：

```
checkpoints/
├── best_model.pt      # 验证损失最优的模型
├── last_model.pt      # 最后一个检查点
├── src_vocab.json     # 源语言词汇表
├── tgt_vocab.json     # 目标语言词汇表
├── config.json        # 训练配置
└── training_log.txt   # 训练日志（CSV格式）
```

## 代码示例（Python API）

```python
from train import train
from translate import Translator
from config import get_small_config
import torch

# 训练
config = get_small_config()
model, src_vocab, tgt_vocab = train(config)

# 翻译
device = torch.device("cpu")
translator = Translator(model, src_vocab, tgt_vocab, config, device)

# 贪婪解码
result = translator.translate("hello", method='greedy')
print(result)  # 你好

# Beam Search
result = translator.translate("good morning", method='beam')
print(result)  # 早上好
```

## 注意事项

1. **示例数据集较小**（~100条），适合验证代码正确性，翻译质量有限。生产环境请使用大规模平行语料（如 WMT、OPUS 等）。
2. **字符级中文分词**：当前中文采用字符级分词。如需更好效果，推荐集成 `jieba` 等分词库。
3. **内存需求**：小型模型在 CPU 上即可运行；大型模型建议使用 GPU。
4. **训练时间**：小型模型在 CPU 上训练 100 epoch 约需 1-2 分钟（示例数据）。

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
