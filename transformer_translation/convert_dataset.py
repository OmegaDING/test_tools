"""
将 wmt_zh_en_training_corpus.csv 转换为 train.py 可直接使用的 TSV 格式

原始格式:
  CSV，列0=中文（已空格分词），列1=英文
  首行为标题 "0,1"

输出格式（英译中，与 train.py 默认方向一致）:
  每行: 英文\t中文（去掉中文内部空格，还原字符级格式）
  文件: wmt_en_zh.tsv

用法:
  python convert_dataset.py                    # 默认取前 500,000 条
  python convert_dataset.py --max 100000       # 只取前 10 万条
  python convert_dataset.py --max -1           # 全量转换（约 2400 万条，耗时较长）
  python convert_dataset.py --max 200000 --output my_data.tsv
"""

import csv
import os
import sys
import argparse
import time


def convert(
    input_path: str,
    output_path: str,
    max_rows: int = 500_000,
    min_en_len: int = 2,
    max_en_len: int = 100,
    min_zh_len: int = 1,
    max_zh_len: int = 100,
    skip_header: bool = True,
):
    """
    转换数据集

    Args:
        input_path:   原始 CSV 文件路径
        output_path:  输出 TSV 文件路径
        max_rows:     最多保留多少行（-1 表示全部）
        min_en_len:   英文最少词数（按空格分）
        max_en_len:   英文最多词数
        min_zh_len:   中文最少字符数（去空格后）
        max_zh_len:   中文最多字符数
        skip_header:  是否跳过首行标题
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    kept = 0
    skipped = 0
    total_read = 0
    start = time.time()

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"最大行数: {'全量' if max_rows == -1 else f'{max_rows:,}'}")
    print(f"开始转换...\n")

    with open(input_path, 'r', encoding='utf-8', errors='replace', newline='') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.reader(fin)

        if skip_header:
            try:
                next(reader)  # 跳过标题行
            except StopIteration:
                print("文件为空！")
                return

        for row in reader:
            total_read += 1

            # 进度提示
            if total_read % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  已读取 {total_read:>10,} 行 | 保留 {kept:>8,} | 耗时 {elapsed:.1f}s")

            # 列数检查
            if len(row) < 2:
                skipped += 1
                continue

            zh_raw = row[0].strip()   # 列0 = 中文（含空格分词）
            en_raw = row[1].strip()   # 列1 = 英文

            if not zh_raw or not en_raw:
                skipped += 1
                continue

            # 中文：去掉分词空格，还原自然字符串
            zh = zh_raw.replace(' ', '')

            # 英文：转小写，保留原始词序
            en = en_raw.lower().strip()

            # 长度过滤（词数/字数）
            en_words = en.split()
            zh_chars = list(zh)

            if not (min_en_len <= len(en_words) <= max_en_len):
                skipped += 1
                continue
            if not (min_zh_len <= len(zh_chars) <= max_zh_len):
                skipped += 1
                continue

            # 写入 TSV：英文\t中文
            fout.write(f"{en}\t{zh}\n")
            kept += 1

            if max_rows != -1 and kept >= max_rows:
                break

    elapsed = time.time() - start
    print(f"\n转换完成！")
    print(f"  读取行数: {total_read:,}")
    print(f"  保留行数: {kept:,}")
    print(f"  跳过行数: {skipped:,}")
    print(f"  总耗时:   {elapsed:.1f}s")
    print(f"  输出文件: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="转换 WMT 中英数据集为 TSV 格式")
    parser.add_argument('--input', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'wmt_zh_en_training_corpus.csv'),
                        help='输入 CSV 文件路径')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'wmt_en_zh.tsv'),
                        help='输出 TSV 文件路径')
    parser.add_argument('--max', type=int, default=500_000,
                        help='最多保留行数（-1 为全量，默认 500000）')
    parser.add_argument('--max_en_len', type=int, default=80,
                        help='英文最多词数（默认 80）')
    parser.add_argument('--max_zh_len', type=int, default=80,
                        help='中文最多字符数（默认 80）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        input_path=args.input,
        output_path=args.output,
        max_rows=args.max,
        max_en_len=args.max_en_len,
        max_zh_len=args.max_zh_len,
    )

    print(f"""
========================================
训练命令（使用转换后的数据集）：

  # 中型模型（推荐，平衡速度与效果）
  cd transformer_translation
  python train.py \\
      --data wmt_en_zh.tsv \\
      --src_lang en \\
      --tgt_lang zh \\
      --size medium \\
      --epochs 30 \\
      --batch_size 64 \\
      --output_dir checkpoints_wmt

  # 小型模型（快速验证）
  python train.py \\
      --data wmt_en_zh.tsv \\
      --src_lang en \\
      --tgt_lang zh \\
      --size small \\
      --epochs 10 \\
      --batch_size 32 \\
      --output_dir checkpoints_wmt

  # 训练完成后翻译
  python translate.py \\
      --model checkpoints_wmt/best_model.pt \\
      --src_vocab checkpoints_wmt/src_vocab.json \\
      --tgt_vocab checkpoints_wmt/tgt_vocab.json \\
      --config checkpoints_wmt/config.json \\
      --text "hello world"
========================================
""")
