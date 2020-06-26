"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

import models
from pytorch_transformers import BertConfig, BertTokenizer, BertForMedical, BertWithCAMLForMedical, BertTinyParallel3WithCAMLForMedical, BertTinyParallel4WithCAMLForMedical
from constants import *
import datasets
import persistence
import numpy as np
from transformers import BertModel


def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    Y = len(dicts['ind2c'])
    if args.model == "rnn":
        model = models.VanillaRNN(Y, args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout, code_emb=args.code_emb)
    elif args.model == "logreg":
        model = models.BOWPool(Y, args.embed_file, args.lmbda, args.gpu, dicts, args.pool, args.embed_size, args.dropout, args.code_emb)
    elif args.model == 'bert':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-base-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-base-uncased-vocab.txt', do_lower_case=True)

        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model
        if args.from_scratch and not args.pretrain:
            model = BertForMedical(config=config)
        elif args.pretrain:
            model = BertForMedical.from_pretrained(args.pretrain_ckpt_dir)
        else:
            model = BertForMedical.from_pretrained('./pretrained_weights/bert-base-uncased-pytorch_model.bin', config=config)
    elif args.model == 'biobert':
        config = BertConfig.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/bert_config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=False)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/vocab.txt', do_lower_case=False)

        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model
        if args.from_scratch and not args.pretrain:
            model = BertForMedical(config=config)
        elif args.pretrain:
            model = BertForMedical.from_pretrained(args.pretrain_ckpt_dir)
        else:
            model = BertForMedical.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/pytorch_model.bin', config=config)
    elif args.model == 'bert-tiny':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-tiny-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)

        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model
        if args.from_scratch and not args.pretrain:
            model = BertForMedical(config=config)
        elif args.pretrain:
            model = BertForMedical.from_pretrained(args.pretrain_ckpt_dir)
        else:
            model = BertForMedical.from_pretrained('./pretrained_weights/bert-tiny-uncased-pytorch_model.bin', config=config)
    elif args.model == 'bert-caml':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-base-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-base-uncased-vocab.txt', do_lower_case=True)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.embed_size = args.embed_size
        config.embed_file = args.embed_file
        config.dicts = dicts
        config.model = args.model
        if args.from_scratch:
            model = BertWithCAMLForMedical(config=config)
        else:
            model = BertWithCAMLForMedical.from_pretrained('./pretrained_weights/bert-base-uncased-pytorch_model.bin', config=config)
    elif args.model == 'bert-small-caml':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-small-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-small-uncased-vocab.txt', do_lower_case=True)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.embed_size = args.embed_size
        config.embed_file = args.embed_file
        config.dicts = dicts
        config.model = args.model
        if args.from_scratch:
            model = BertWithCAMLForMedical(config=config)
        else:
            model = BertWithCAMLForMedical.from_pretrained('./pretrained_weights/bert-small-uncased-pytorch_model.bin', config=config)
    elif args.model == 'bert-tiny-caml':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-tiny-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.embed_size = args.embed_size
        config.embed_file = args.embed_file
        config.dicts = dicts
        config.model = args.model
        if args.from_scratch:
            model = BertWithCAMLForMedical(config=config)
        else:
            model = BertWithCAMLForMedical.from_pretrained('./pretrained_weights/bert-tiny-uncased-pytorch_model.bin', config=config)
    elif args.model == 'bert-tiny-parallel3-caml':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-tiny-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.embed_size = args.embed_size
        config.embed_file = args.embed_file
        config.dicts = dicts
        config.model = args.model
        if args.from_scratch:
            model = BertTinyParallel3WithCAMLForMedical(config=config)
        else:
            model = BertTinyParallel3WithCAMLForMedical.from_pretrained('./pretrained_weights/bert-tiny-uncased-pytorch_model.bin', config=config)
    elif args.model == 'bert-tiny-parallel4-caml':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-tiny-uncased-config.json')
        config.Y = int(args.Y)
        config.gpu = args.gpu
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.embed_size = args.embed_size
        config.embed_file = args.embed_file
        config.dicts = dicts
        config.model = args.model
        if args.from_scratch:
            model = BertTinyParallel4WithCAMLForMedical(config=config)
        else:
            model = BertTinyParallel4WithCAMLForMedical.from_pretrained('./pretrained_weights/bert-tiny-uncased-pytorch_model.bin', config=config)

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu:
        model.cuda()
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers, 
                  args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "lmbda", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

