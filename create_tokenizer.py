import argparse
from tokenizers import BertWordPieceTokenizer

# train_corpus_file = './mimicdata/bio-mimic3/train_50.csv'
# dev_corpus_file = './mimicdata/bio-mimic3/dev_50.csv'
# test_corpus_file = './mimicdata/bio-mimic3/test_50.csv'

train_corpus_file = './mimicdata/mimic3/train_full.csv'
dev_corpus_file = './mimicdata/mimic3/dev_full.csv'
test_corpus_file = './mimicdata/mimic3/test_full.csv'

limit_alphabet = 100
vocab_size = 100000

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False, # Must be False if cased model
    lowercase=True,
    wordpieces_prefix="##",
)

tokenizer.train(
    files=[train_corpus_file, dev_corpus_file, test_corpus_file],
    limit_alphabet=limit_alphabet,
    vocab_size=vocab_size,
    min_frequency=1,
)

# tokenizer.save("./tokenizers", "bert-tiny-mimic3-50-{}-limit-{}".format(limit_alphabet, vocab_size))
tokenizer.save("./tokenizers", "bert-tiny-mimic3-full-{}-limit-{}".format(limit_alphabet, vocab_size))
