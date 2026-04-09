"""
翻译推理脚本
支持：
  - 贪婪解码（Greedy Decoding）
  - Beam Search 解码
  - 交互式命令行翻译
  - 批量文件翻译
"""

import os
import sys
import math
import argparse
from typing import List, Tuple, Optional

import torch

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_model import Transformer
from vocabulary import Vocabulary, Tokenizer, PAD_IDX, BOS_IDX, EOS_IDX
from config import TranslationConfig


# ============================================================
# 解码器
# ============================================================

class Translator:
    """翻译器：支持贪婪解码和 Beam Search"""
    
    def __init__(
        self,
        model: Transformer,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        config: TranslationConfig,
        device: torch.device,
    ):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = device
        self.tokenizer = Tokenizer()
        self.model.eval()
    
    def _tokenize_src(self, text: str) -> List[str]:
        """对源语言文本分词"""
        if self.config.src_lang == 'zh':
            return self.tokenizer.tokenize_zh(text)
        return self.tokenizer.tokenize_en(text)
    
    def _src_to_tensor(self, text: str) -> torch.Tensor:
        """将源语言文本转换为索引张量（含BOS/EOS）"""
        tokens = self._tokenize_src(text)
        indices = [BOS_IDX] + self.src_vocab.encode(tokens) + [EOS_IDX]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def _decode_output(self, indices: List[int]) -> str:
        """将输出索引解码为文本"""
        tokens = self.tgt_vocab.decode(indices, remove_special=True)
        if self.config.tgt_lang == 'zh':
            # 中文直接拼接
            return ''.join(tokens)
        else:
            # 英文用空格连接
            return ' '.join(tokens)
    
    @torch.no_grad()
    def greedy_decode(self, src_text: str) -> Tuple[str, List[int]]:
        """
        贪婪解码
        
        Args:
            src_text: 源语言文本
        
        Returns:
            (translated_text, token_indices)
        """
        src = self._src_to_tensor(src_text)
        
        # 编码源序列
        enc_output, src_mask = self.model.encode(src, pad_idx=PAD_IDX)
        
        # 初始化解码序列（只有BOS）
        tgt = torch.tensor([[BOS_IDX]], dtype=torch.long).to(self.device)
        
        output_indices = []
        max_len = self.config.max_translate_len
        
        for _ in range(max_len):
            # 解码一步
            logits = self.model.decode(tgt, enc_output, src_mask, pad_idx=PAD_IDX)
            
            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token = next_token_logits.argmax(dim=-1)  # (1,)
            
            token_id = next_token.item()
            
            if token_id == EOS_IDX:
                break
            
            output_indices.append(token_id)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
        
        translated = self._decode_output(output_indices)
        return translated, output_indices
    
    @torch.no_grad()
    def beam_search(self, src_text: str, beam_size: int = 4, length_penalty: float = 0.6) -> Tuple[str, List[int]]:
        """
        Beam Search 解码
        
        Args:
            src_text: 源语言文本
            beam_size: Beam 宽度
            length_penalty: 长度惩罚系数（>1 鼓励长句，<1 鼓励短句）
        
        Returns:
            (best_translation, best_indices)
        """
        src = self._src_to_tensor(src_text)
        max_len = self.config.max_translate_len
        
        # 编码源序列
        enc_output, src_mask = self.model.encode(src, pad_idx=PAD_IDX)
        
        # 将编码输出扩展为 beam_size 份
        enc_output = enc_output.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)
        
        # 初始化 beam：每个 beam 包含 (累积log概率, token序列)
        beams = [(0.0, [BOS_IDX])]
        completed = []
        
        for step in range(max_len):
            candidates = []
            
            # 对每个 beam 进行扩展
            active_beams = [(score, seq) for score, seq in beams if seq[-1] != EOS_IDX]
            if not active_beams:
                break
            
            # 构建当前 beam 的输入张量
            tgt_seqs = torch.tensor(
                [seq for _, seq in active_beams],
                dtype=torch.long
            ).to(self.device)
            
            n_active = tgt_seqs.size(0)
            
            # 解码
            logits = self.model.decode(
                tgt_seqs,
                enc_output[:n_active],
                src_mask[:n_active],
                pad_idx=PAD_IDX
            )
            
            # 取最后位置的 log 概率
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (n_active, vocab)
            
            for beam_idx, (score, seq) in enumerate(active_beams):
                beam_log_probs = log_probs[beam_idx]  # (vocab,)
                
                # 取 top-k 候选
                topk_log_probs, topk_ids = beam_log_probs.topk(beam_size)
                
                for log_prob, token_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_score = score + log_prob
                    new_seq = seq + [token_id]
                    
                    if token_id == EOS_IDX:
                        # 长度惩罚
                        seq_len = len(new_seq) - 1  # 不含BOS
                        penalty = ((5 + seq_len) / 6) ** length_penalty
                        completed.append((new_score / penalty, new_seq))
                    else:
                        candidates.append((new_score, new_seq))
            
            # 保留最优的 beam_size 个候选
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]
            
            if not beams:
                break
        
        # 如果没有完成的序列，取当前最优 beam
        if not completed:
            for score, seq in beams:
                seq_len = max(len(seq) - 1, 1)
                penalty = ((5 + seq_len) / 6) ** length_penalty
                completed.append((score / penalty, seq))
        
        # 选择得分最高的序列
        completed.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seq = completed[0]
        
        # 去除 BOS 和 EOS
        best_indices = [t for t in best_seq if t not in (BOS_IDX, EOS_IDX)]
        
        translated = self._decode_output(best_indices)
        return translated, best_indices
    
    def translate(self, src_text: str, method: str = 'beam') -> str:
        """
        翻译主接口
        
        Args:
            src_text: 源语言文本
            method: 解码方法 ('greedy' 或 'beam')
        
        Returns:
            翻译结果文本
        """
        src_text = src_text.strip()
        if not src_text:
            return ""
        
        try:
            if method == 'greedy' or self.config.beam_size <= 1:
                result, _ = self.greedy_decode(src_text)
            else:
                result, _ = self.beam_search(
                    src_text,
                    beam_size=self.config.beam_size,
                    length_penalty=self.config.length_penalty
                )
            return result
        except Exception as e:
            return f"[翻译错误: {e}]"
    
    def translate_batch(self, texts: List[str], method: str = 'beam') -> List[str]:
        """批量翻译"""
        return [self.translate(text, method) for text in texts]


# ============================================================
# 模型加载
# ============================================================

def load_translator(
    model_path: str,
    src_vocab_path: str,
    tgt_vocab_path: str,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Translator:
    """
    加载训练好的翻译模型
    
    Args:
        model_path: 模型权重文件路径 (.pt)
        src_vocab_path: 源语言词汇表路径
        tgt_vocab_path: 目标语言词汇表路径
        config_path: 配置文件路径（可选）
        device: 运算设备（默认自动检测）
    
    Returns:
        Translator 对象
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载词汇表
    src_vocab = Vocabulary.load(src_vocab_path)
    tgt_vocab = Vocabulary.load(tgt_vocab_path)
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        config = TranslationConfig.load(config_path)
    else:
        # 从检查点恢复配置
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = TranslationConfig()
        if 'config' in checkpoint:
            for k, v in checkpoint['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)
    
    # 构建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        max_len=config.max_len,
        dropout=0.0,  # 推理时不使用dropout
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
    print(f"模型加载成功 (训练轮数: {epoch}, 验证损失: {val_loss_str})")
    
    return Translator(model, src_vocab, tgt_vocab, config, device)


# ============================================================
# 交互式翻译
# ============================================================

def interactive_translate(translator: Translator, method: str = 'beam'):
    """交互式翻译模式"""
    src_lang_name = {'en': '英语', 'zh': '中文'}.get(translator.config.src_lang, translator.config.src_lang)
    tgt_lang_name = {'en': '英语', 'zh': '中文'}.get(translator.config.tgt_lang, translator.config.tgt_lang)
    
    print(f"\n{'='*60}")
    print(f"  交互式翻译 ({src_lang_name} → {tgt_lang_name})")
    print(f"  解码方式: {'Beam Search' if method == 'beam' else '贪婪解码'}")
    print(f"  输入 'quit' 或 'exit' 退出")
    print(f"{'='*60}\n")
    
    while True:
        try:
            src_text = input(f"[{src_lang_name}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出翻译。")
            break
        
        if src_text.lower() in ('quit', 'exit', 'q', '退出'):
            print("退出翻译。")
            break
        
        if not src_text:
            continue
        
        # 支持运行时切换解码方式
        if src_text.lower() == '--greedy':
            method = 'greedy'
            print("切换到贪婪解码")
            continue
        if src_text.lower() == '--beam':
            method = 'beam'
            print(f"切换到 Beam Search (beam_size={translator.config.beam_size})")
            continue
        
        result = translator.translate(src_text, method=method)
        print(f"[{tgt_lang_name}] > {result}\n")


# ============================================================
# 命令行接口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer 翻译推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式翻译（使用默认检查点）
  python translate.py

  # 翻译单句
  python translate.py --text "hello world"

  # 批量翻译文件
  python translate.py --input input.txt --output output.txt

  # 指定模型路径
  python translate.py --model checkpoints/best_model.pt --text "good morning"

  # 使用贪婪解码
  python translate.py --method greedy --text "thank you"
        """
    )
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='模型文件路径')
    parser.add_argument('--src_vocab', type=str, default='checkpoints/src_vocab.json',
                        help='源语言词汇表路径')
    parser.add_argument('--tgt_vocab', type=str, default='checkpoints/tgt_vocab.json',
                        help='目标语言词汇表路径')
    parser.add_argument('--config', type=str, default='checkpoints/config.json',
                        help='配置文件路径')
    parser.add_argument('--text', type=str, default=None,
                        help='要翻译的文本（单句）')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径（每行一个句子）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--method', type=str, default='beam',
                        choices=['greedy', 'beam'],
                        help='解码方法')
    parser.add_argument('--beam_size', type=int, default=None,
                        help='Beam Search 宽度')
    parser.add_argument('--interactive', action='store_true',
                        help='启动交互式翻译模式')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 找不到模型文件 '{args.model}'")
        print("请先运行 train.py 训练模型，或指定正确的模型路径。")
        sys.exit(1)
    
    if not os.path.exists(args.src_vocab):
        print(f"错误: 找不到源词汇表 '{args.src_vocab}'")
        sys.exit(1)
    
    if not os.path.exists(args.tgt_vocab):
        print(f"错误: 找不到目标词汇表 '{args.tgt_vocab}'")
        sys.exit(1)
    
    # 设置设备
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 加载翻译器
    translator = load_translator(
        model_path=args.model,
        src_vocab_path=args.src_vocab,
        tgt_vocab_path=args.tgt_vocab,
        config_path=args.config if os.path.exists(args.config) else None,
        device=device,
    )
    
    # 覆盖 beam_size
    if args.beam_size is not None:
        translator.config.beam_size = args.beam_size
    
    # 执行翻译
    if args.text:
        # 单句翻译
        result = translator.translate(args.text, method=args.method)
        print(f"输入: {args.text}")
        print(f"翻译: {result}")
    
    elif args.input:
        # 文件批量翻译
        if not os.path.exists(args.input):
            print(f"错误: 找不到输入文件 '{args.input}'")
            sys.exit(1)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"正在翻译 {len(lines)} 个句子...")
        results = translator.translate_batch(lines, method=args.method)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result + '\n')
            print(f"翻译完成，结果已保存到: {args.output}")
        else:
            for src, tgt in zip(lines, results):
                print(f"  {src}  →  {tgt}")
    
    else:
        # 默认进入交互式模式
        interactive_translate(translator, method=args.method)


# ============================================================
# 快速演示（不依赖已训练模型，直接从内置数据训练后翻译）
# ============================================================

def quick_demo():
    """快速演示：自动训练 + 翻译"""
    print("=" * 60)
    print("Transformer 翻译模型 - 快速演示")
    print("=" * 60)
    
    # 导入训练模块
    from train import train
    from config import get_small_config
    
    # 使用小型配置快速训练
    config = get_small_config()
    config.num_epochs = 80
    config.output_dir = 'checkpoints_demo'
    config.src_vocab_path = 'checkpoints_demo/src_vocab.json'
    config.tgt_vocab_path = 'checkpoints_demo/tgt_vocab.json'
    config.best_model_path = 'checkpoints_demo/best_model.pt'
    config.last_model_path = 'checkpoints_demo/last_model.pt'
    config.log_file = 'checkpoints_demo/training_log.txt'
    config.early_stopping_patience = 15
    
    print("\n正在训练模型（使用内置示例数据）...")
    model, src_vocab, tgt_vocab = train(config)
    
    # 加载翻译器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = Translator(model, src_vocab, tgt_vocab, config, device)
    
    # 测试翻译
    test_sentences = [
        "hello",
        "good morning",
        "thank you",
        "how are you",
        "i love you",
        "goodbye",
        "happy birthday",
        "i am hungry",
        "do not worry",
        "never give up",
    ]
    
    print("\n" + "=" * 60)
    print("翻译测试结果（英文 → 中文）")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        # 贪婪解码
        greedy_result = translator.translate(sentence, method='greedy')
        # Beam Search
        beam_result = translator.translate(sentence, method='beam')
        
        print(f"\n[{i:2}] 原文: {sentence}")
        print(f"     贪婪: {greedy_result}")
        print(f"     Beam: {beam_result}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("运行 'python translate.py' 进入交互式翻译模式")
    print("=" * 60)


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 无参数时检查是否存在训练好的模型
        if not os.path.exists('checkpoints/best_model.pt'):
            print("未找到已训练的模型，启动快速演示模式（训练+翻译）...")
            quick_demo()
        else:
            # 进入交互式翻译
            main()
    else:
        main()
