import argparse
from tokenizers import BertWordPieceTokenizer

train_corpus_file = './mimicdata/bio-mimic3/train_50.csv'
dev_corpus_file = './mimicdata/bio-mimic3/dev_50.csv'
test_corpus_file = './mimicdata/bio-mimic3/test_50.csv'

# train_corpus_file = './mimicdata/mimic3/train_50.csv'
# dev_corpus_file = './mimicdata/mimic3/dev_50.csv'
# test_corpus_file = './mimicdata/mimic3/test_50.csv'

limit_alphabet = 6000
vocab_size = 10000

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False, # Must be False if cased model
    lowercase=True,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[train_corpus_file, dev_corpus_file, test_corpus_file],
    limit_alphabet=limit_alphabet,
    vocab_size=vocab_size
)

tokenizer.save("./tokenizers", "bio-mimic3-{}-limit-{}".format(limit_alphabet, vocab_size))
# tokenizer.save("./tokenizers", "mimic3-{}-limit-{}".format(limit_alphabet, vocab_size))
