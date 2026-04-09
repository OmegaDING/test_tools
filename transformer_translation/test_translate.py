"""快速翻译测试脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from translate import load_translator
import torch

device = torch.device('cpu')
base = os.path.join(os.path.dirname(__file__), 'checkpoints')

translator = load_translator(
    model_path=os.path.join(base, 'best_model.pt'),
    src_vocab_path=os.path.join(base, 'src_vocab.json'),
    tgt_vocab_path=os.path.join(base, 'tgt_vocab.json'),
    config_path=os.path.join(base, 'config.json'),
    device=device
)

tests = [
    'hello',
    'good morning',
    'thank you',
    'how are you',
    'i love you',
    'goodbye',
    'happy birthday',
    'i am hungry',
    'do not worry',
    'never give up',
    'i am fine',
    'good night',
    'you are welcome',
    'i am from china',
    'the weather is nice today',
]

print()
print(f"{'原文':<35} {'贪婪解码':<20} {'Beam Search'}")
print("-" * 75)
for s in tests:
    g = translator.translate(s, method='greedy')
    b = translator.translate(s, method='beam')
    print(f"  {s:<33} {g:<20} {b}")
