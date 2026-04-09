"""
词汇表构建与文本处理工具
支持简单的字符级和词级分词
"""

import re
import json
import os
from collections import Counter
from typing import List, Dict, Optional


# 特殊标记
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'  # Beginning of Sentence
EOS_TOKEN = '<eos>'  # End of Sentence
UNK_TOKEN = '<unk>'  # Unknown

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """词汇表类"""
    
    def __init__(self, name: str = "vocab"):
        self.name = name
        self.token2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            BOS_TOKEN: BOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}
        self.token_freq: Counter = Counter()
    
    def __len__(self):
        return len(self.token2idx)
    
    def __contains__(self, token):
        return token in self.token2idx
    
    def add_token(self, token: str) -> int:
        """添加单个token到词汇表"""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def build_from_texts(self, texts: List[List[str]], min_freq: int = 1):
        """从文本列表构建词汇表
        
        Args:
            texts: 分词后的文本列表（每个元素是一个token列表）
            min_freq: 最小词频阈值
        """
        for tokens in texts:
            self.token_freq.update(tokens)
        
        # 按频率排序后添加到词汇表
        for token, freq in sorted(self.token_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq:
                self.add_token(token)
        
        print(f"[{self.name}] 词汇表大小: {len(self)}")
        return self
    
    def encode(self, tokens: List[str]) -> List[int]:
        """将token列表转为索引列表"""
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]
    
    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """将索引列表转为token列表"""
        tokens = [self.idx2token.get(i, UNK_TOKEN) for i in indices]
        if remove_special:
            tokens = [t for t in tokens if t not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)]
        return tokens
    
    def save(self, path: str):
        """保存词汇表到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'name': self.name,
            'token2idx': self.token2idx,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"词汇表已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """从文件加载词汇表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(data['name'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        print(f"词汇表已加载: {path} (大小: {len(vocab)})")
        return vocab


class Tokenizer:
    """分词器"""
    
    @staticmethod
    def tokenize_en(text: str) -> List[str]:
        """英文分词（简单空格分词 + 标点处理）"""
        text = text.lower().strip()
        # 在标点前后添加空格
        text = re.sub(r"([.!?,;:\"'()\[\]{}])", r" \1 ", text)
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    @staticmethod
    def tokenize_zh(text: str) -> List[str]:
        """中文分词（字符级，每个汉字作为一个token）"""
        text = text.strip()
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f':
                # 中文字符逐字分割
                tokens.append(char)
            elif char.isspace():
                i += 1
                continue
            else:
                # 非中文字符按词处理
                j = i
                while j < len(text) and not ('\u4e00' <= text[j] <= '\u9fff') and not text[j].isspace():
                    j += 1
                word = text[i:j]
                if word:
                    tokens.append(word.lower())
                i = j
                continue
            i += 1
        return tokens
    
    @staticmethod
    def tokenize_auto(text: str) -> List[str]:
        """自动检测语言并分词"""
        # 检测是否包含中文
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        if has_chinese:
            return Tokenizer.tokenize_zh(text)
        else:
            return Tokenizer.tokenize_en(text)


def build_vocab_from_pairs(
    pairs: List[tuple], 
    src_min_freq: int = 1, 
    tgt_min_freq: int = 1,
    src_lang: str = "en",
    tgt_lang: str = "zh"
) -> tuple:
    """
    从句对数据构建源语言和目标语言词汇表
    
    Args:
        pairs: [(src_sentence, tgt_sentence), ...] 句对列表
        src_min_freq: 源语言最小词频
        tgt_min_freq: 目标语言最小词频
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
    
    Returns:
        (src_vocab, tgt_vocab)
    """
    src_texts = []
    tgt_texts = []
    
    tokenizer = Tokenizer()
    
    for src, tgt in pairs:
        if src_lang == 'zh':
            src_texts.append(tokenizer.tokenize_zh(src))
        else:
            src_texts.append(tokenizer.tokenize_en(src))
        
        if tgt_lang == 'zh':
            tgt_texts.append(tokenizer.tokenize_zh(tgt))
        else:
            tgt_texts.append(tokenizer.tokenize_en(tgt))
    
    src_vocab = Vocabulary(f"{src_lang}_vocab")
    src_vocab.build_from_texts(src_texts, min_freq=src_min_freq)
    
    tgt_vocab = Vocabulary(f"{tgt_lang}_vocab")
    tgt_vocab.build_from_texts(tgt_texts, min_freq=tgt_min_freq)
    
    return src_vocab, tgt_vocab


# ============================================================
# 示例训练数据（英译中）
# ============================================================
SAMPLE_EN_ZH_PAIRS = [
    ("hello", "你好"),
    ("good morning", "早上好"),
    ("good evening", "晚上好"),
    ("good night", "晚安"),
    ("thank you", "谢谢"),
    ("thank you very much", "非常感谢"),
    ("you are welcome", "不客气"),
    ("sorry", "对不起"),
    ("excuse me", "打扰一下"),
    ("how are you", "你好吗"),
    ("i am fine", "我很好"),
    ("nice to meet you", "很高兴认识你"),
    ("what is your name", "你叫什么名字"),
    ("my name is john", "我的名字是约翰"),
    ("where are you from", "你来自哪里"),
    ("i am from china", "我来自中国"),
    ("i love you", "我爱你"),
    ("i miss you", "我想你"),
    ("happy birthday", "生日快乐"),
    ("merry christmas", "圣诞快乐"),
    ("happy new year", "新年快乐"),
    ("good luck", "祝你好运"),
    ("take care", "保重"),
    ("goodbye", "再见"),
    ("see you later", "再见"),
    ("see you tomorrow", "明天见"),
    ("i am hungry", "我饿了"),
    ("i am thirsty", "我渴了"),
    ("i am tired", "我累了"),
    ("i am happy", "我很开心"),
    ("i am sad", "我很难过"),
    ("i am busy", "我很忙"),
    ("i am free", "我有空"),
    ("i understand", "我明白"),
    ("i do not understand", "我不明白"),
    ("please speak slowly", "请说慢一点"),
    ("can you help me", "你能帮我吗"),
    ("of course", "当然"),
    ("no problem", "没问题"),
    ("wait a moment", "等一下"),
    ("how much is this", "这个多少钱"),
    ("it is too expensive", "太贵了"),
    ("can you give me a discount", "可以打折吗"),
    ("where is the bathroom", "洗手间在哪里"),
    ("i need a doctor", "我需要医生"),
    ("call the police", "叫警察"),
    ("i am lost", "我迷路了"),
    ("turn left", "向左转"),
    ("turn right", "向右转"),
    ("go straight", "直走"),
    ("stop here", "在这里停"),
    ("the weather is nice today", "今天天气很好"),
    ("it is raining", "下雨了"),
    ("it is snowing", "下雪了"),
    ("it is hot today", "今天很热"),
    ("it is cold today", "今天很冷"),
    ("what time is it", "现在几点"),
    ("it is three o clock", "现在三点"),
    ("let us go", "我们走吧"),
    ("wait for me", "等等我"),
    ("follow me", "跟我来"),
    ("be careful", "小心"),
    ("do not worry", "别担心"),
    ("everything will be fine", "一切都会好的"),
    ("i believe in you", "我相信你"),
    ("you can do it", "你能做到"),
    ("well done", "做得好"),
    ("congratulations", "恭喜"),
    ("i am proud of you", "我为你骄傲"),
    ("keep going", "继续加油"),
    ("never give up", "永不放弃"),
    ("dream big", "要有远大的梦想"),
    ("work hard", "努力工作"),
    ("study hard", "努力学习"),
    ("practice makes perfect", "熟能生巧"),
    ("time is money", "时间就是金钱"),
    ("knowledge is power", "知识就是力量"),
    ("actions speak louder than words", "行动胜于言辞"),
    ("where there is a will there is a way", "有志者事竟成"),
    ("the early bird catches the worm", "早起的鸟儿有虫吃"),
    ("i would like a cup of coffee", "我想要一杯咖啡"),
    ("i would like a cup of tea", "我想要一杯茶"),
    ("the food is delicious", "食物很美味"),
    ("i like chinese food", "我喜欢中国食物"),
    ("i like to travel", "我喜欢旅行"),
    ("i like reading books", "我喜欢读书"),
    ("i like listening to music", "我喜欢听音乐"),
    ("i like watching movies", "我喜欢看电影"),
    ("i like playing sports", "我喜欢运动"),
    ("please give me the menu", "请给我菜单"),
    ("i would like to order", "我想点菜"),
    ("the bill please", "请结账"),
    ("do you speak english", "你会说英语吗"),
    ("i speak a little chinese", "我会说一点中文"),
    ("can i take a photo here", "我可以在这里拍照吗"),
    ("this is beautiful", "这很漂亮"),
    ("i like this", "我喜欢这个"),
    ("i do not like this", "我不喜欢这个"),
    ("this is interesting", "这很有趣"),
    ("today is monday", "今天是星期一"),
    ("tomorrow is tuesday", "明天是星期二"),
]


if __name__ == "__main__":
    # 测试词汇表构建
    print("构建词汇表...")
    src_vocab, tgt_vocab = build_vocab_from_pairs(
        SAMPLE_EN_ZH_PAIRS,
        src_lang='en',
        tgt_lang='zh'
    )
    
    print(f"\n英文词汇表大小: {len(src_vocab)}")
    print(f"中文词汇表大小: {len(tgt_vocab)}")
    
    # 测试编解码
    tokenizer = Tokenizer()
    src_tokens = tokenizer.tokenize_en("hello how are you")
    tgt_tokens = tokenizer.tokenize_zh("你好吗")
    
    print(f"\n英文分词: {src_tokens}")
    print(f"中文分词: {tgt_tokens}")
    
    print(f"英文编码: {src_vocab.encode(src_tokens)}")
    print(f"中文编码: {tgt_vocab.encode(tgt_tokens)}")
