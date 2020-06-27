"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

import csv
import argparse
import os
import numpy as np
import operator
import random
import sys
import time
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler

from constants import *
import datasets
import evaluation
import interpret
import persistence
import models as models
import tools as tools
from pytorch_transformers import BertConfig, BertTokenizer, BertForMaskedLM


BERT_MODEL_LIST = ['bert', 'biobert', 'bert-caml', 'bert-samll-caml', 'bert-tiny-caml', 'bert-tiny', 'bert-tiny-parallel-caml']


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):
    start = time.time()
    if args.pretrain:
        pretrain(args, args.data_path)

    args, model, optimizer, params, dicts, scheduler, labels_weight = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts, scheduler, labels_weight)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))


def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    #load vocab and other lookups
    desc_embed = args.lmbda > 0
    print("loading lookups...")
    dicts = datasets.load_lookups(args, desc_embed=desc_embed)

    model = tools.pick_model(args, dicts)
    print(model)

    if not args.test_model:
        if args.model in BERT_MODEL_LIST:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=args.weight_decay, lr=args.lr)
            length = datasets.data_length(args.data_path, args.version)
            t_total = length // args.batch_size * args.n_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, \
                                                        num_warmup_steps=args.warmup_steps, \
                                                        num_training_steps=t_total, \
                                                       )
            def get_label_distribution(filename, dicts):
                ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
                if args.Y == 'full':
                    labels_idx = [1e-15] * 8921
                else:
                    labels_idx = [1e-15] * int(args.Y)
                with open(filename, 'r') as infile:
                    r = csv.reader(infile)
                    # header
                    next(r)
                    for row in r:
                        for l in row[3].split(';'):
                            if l in c2ind.keys():
                                code = int(c2ind[l])
                                labels_idx[code] += 1
                max_val = max(labels_idx)
                return max_val / np.array(labels_idx)
            if args.pos:
                labels_weight = get_label_distribution(args.data_path, dicts)
            else:
                labels_weight = None
        else:
            optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
            scheduler = None
            labels_weight = None
    else:
        optimizer = None
        scheduler = None
        labels_weight = None

    params = tools.make_param_dict(args)

    return args, model, optimizer, params, dicts, scheduler, labels_weight


def train_epochs(args, model, optimizer, params, dicts, scheduler, labels_weight):
    """
        Main loop. does train and test
    """

    dev_acc_macro, dev_prec_macro, dev_rec_macro, dev_f1_macro = 0.0, 0.0, 0.0, 0.0
    dev_acc_micro, dev_prec_micro, dev_rec_micro, dev_f1_micro = 0.0, 0.0, 0.0, 0.0
    dev_rec_at_5, dev_prec_at_5, dev_f1_at_5 = 0.0, 0.0, 0.0
    dev_auc_macro, dev_auc_micro = 0.0, 0.0
    dev_epoch, dev_loss = 0.0, 0.0

    test_acc_macro, test_prec_macro, test_rec_macro, test_f1_macro = 0.0, 0.0, 0.0, 0.0
    test_acc_micro, test_prec_micro, test_rec_micro, test_f1_micro = 0.0, 0.0, 0.0, 0.0
    test_rec_at_5, test_prec_at_5, test_f1_at_5 = 0.0, 0.0, 0.0
    test_auc_macro, test_auc_micro = 0.0, 0.0
    test_epoch, test_loss = 0.0, 0.0

    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None

    # train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        best = False
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
        metrics_all = one_epoch(args, \
                                model, \
                                optimizer, \
                                args.Y, \
                                epoch, \
                                args.n_epochs, \
                                args.batch_size, \
                                args.data_path, \
                                args.version, \
                                test_only, \
                                dicts, \
                                model_dir, \
                                args.samples, \
                                args.gpu, \
                                args.quiet, \
                                scheduler, \
                                labels_weight, \
                               )
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        if metrics_hist_te['auc_micro'][-1] >= test_auc_micro:
            test_acc_macro = metrics_hist_te['acc_macro'][-1]
            test_prec_macro = metrics_hist_te['prec_macro'][-1]
            test_rec_macro = metrics_hist_te['rec_macro'][-1]
            test_f1_macro = metrics_hist_te['f1_macro'][-1]
            test_acc_micro = metrics_hist_te['acc_micro'][-1]
            test_prec_micro = metrics_hist_te['prec_micro'][-1]
            test_rec_micro = metrics_hist_te['rec_micro'][-1]
            test_f1_micro = metrics_hist_te['f1_micro'][-1]
            test_rec_at_5 = metrics_hist_te['rec_at_5'][-1]
            test_prec_at_5 = metrics_hist_te['prec_at_5'][-1]
            test_f1_at_5 = metrics_hist_te['f1_at_5'][-1]
            test_auc_macro = metrics_hist_te['auc_macro'][-1]
            test_auc_micro = metrics_hist_te['auc_micro'][-1]
            test_loss = metrics_hist_te['loss_test'][-1]
            test_epoch = epoch

        if metrics_hist['auc_micro'][-1] >= dev_auc_micro:
            dev_acc_macro = metrics_hist['acc_macro'][-1]
            dev_prec_macro = metrics_hist['prec_macro'][-1]
            dev_rec_macro = metrics_hist['rec_macro'][-1]
            dev_f1_macro = metrics_hist['f1_macro'][-1]
            dev_acc_micro = metrics_hist['acc_micro'][-1]
            dev_prec_micro = metrics_hist['prec_micro'][-1]
            dev_rec_micro = metrics_hist['rec_micro'][-1]
            dev_f1_micro = metrics_hist['f1_micro'][-1]
            dev_rec_at_5 = metrics_hist['rec_at_5'][-1]
            dev_prec_at_5 = metrics_hist['prec_at_5'][-1]
            dev_f1_at_5 = metrics_hist['f1_at_5'][-1]
            dev_auc_macro = metrics_hist['auc_macro'][-1]
            dev_auc_micro = metrics_hist['auc_micro'][-1]
            dev_loss = metrics_hist['loss_dev'][-1]
            dev_epoch = epoch
            best = True

        print()
        print('-'*19 + ' Best (Dev) ' + '-'*19)
        print('Best Epoch: %d' % dev_epoch)
        print()
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (dev_acc_macro, dev_prec_macro,
            dev_rec_macro, dev_f1_macro, dev_auc_macro))
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (dev_acc_micro, dev_prec_micro,
            dev_rec_micro, dev_f1_micro, dev_auc_micro))
        print('rec_at_5: %.4f' % dev_rec_at_5)
        print('prec_at_5: %.4f' % dev_prec_at_5)
        print()
        print('Dev loss: %.4f' % dev_loss)
        print()
        print('-'*51)

        print()
        print('-'*19 + ' Best (Test) ' + '-'*19)
        print('Best Epoch: %d' % test_epoch)
        print()
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (test_acc_macro, test_prec_macro,
            test_rec_macro, test_f1_macro, test_auc_macro))
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (test_acc_micro, test_prec_micro,
            test_rec_micro, test_f1_micro, test_auc_micro))
        print('rec_at_5: %.4f' % test_rec_at_5)
        print('prec_at_5: %.4f' % test_prec_at_5)
        print()
        print('Test loss: %.4f' % test_loss)
        print()
        print('-'*51)

        # save metrics, model, params
        persistence.save_everything(args, \
                                    metrics_hist_all, \
                                    model, \
                                    model_dir, \
                                    params, \
                                    args.criterion, \
                                    evaluate, \
                                    best=best, \
                                   )

        if test_only:
            #we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = tools.pick_model(args, dicts)
    return epoch+1


def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev':
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False


def one_epoch(args, model, optimizer, Y, epoch, n_epochs, batch_size, data_path, version, testing, dicts, model_dir, samples, gpu, quiet, scheduler, labels_weight):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        losses, unseen_code_inds, = train(args, \
                                          model, \
                                          optimizer, \
                                          Y, \
                                          epoch, \
                                          batch_size, \
                                          data_path, \
                                          gpu, \
                                          version, \
                                          dicts, \
                                          quiet, \
                                          scheduler, \
                                          labels_weight, \
                                         )
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
    else:
        loss = np.nan
        if model.lmbda > 0:
            #still need to get unseen code inds
            print("getting set of codes not in training set")
            c2ind = dicts['c2ind']
            unseen_code_inds = set(dicts['ind2c'].keys())
            num_labels = len(dicts['ind2c'])
            with open(data_path, 'r') as f:
                r = csv.reader(f)
                #header
                next(r)
                for row in r:
                    unseen_code_inds = unseen_code_inds.difference(set([c2ind[c] for c in row[3].split(';') if c != '']))
            print("num codes not in train set: %d" % len(unseen_code_inds))
        else:
            unseen_code_inds = set()

    fold = 'test' if version == 'mimic2' else 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True
        quiet = False

    # test on dev
    metrics = test(args, \
                   model, \
                   Y, \
                   epoch, \
                   data_path, \
                   fold, \
                   gpu, \
                   version, \
                   unseen_code_inds, \
                   dicts, \
                   samples, \
                   model_dir, \
                   testing, \
                  )

    # if testing or epoch == n_epochs - 1:
    print("\nevaluating on test")
    metrics_te = test(args, \
                      model, \
                      Y, \
                      epoch, \
                      data_path, \
                      "test", \
                      gpu, \
                      version, \
                      unseen_code_inds, \
                      dicts, \
                      samples, \
                      model_dir, \
                      True, \
                     )
    # else:
    #     metrics_te = defaultdict(float)
    #     fpr_te = defaultdict(lambda: [])
    #     tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(args, model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet, scheduler, labels_weight):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])

    losses = []
    #how often to print some info to stdout
    print_every = 25

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    unseen_code_inds = set(ind2c.keys())
    desc_embed = model.lmbda > 0

    if args.model == 'bert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-base-uncased-vocab.txt', do_lower_case=True)
    elif args.model == 'biobert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=False)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/vocab.txt', do_lower_case=False)
    elif args.model == 'bert-tiny':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)
    else:
        bert_tokenizer = None

    model.train()
    model.zero_grad()
    gen = datasets.data_generator(data_path, dicts, batch_size, num_labels,
            version=version, desc_embed=desc_embed, bert_tokenizer=bert_tokenizer, max_seq_length=args.max_sequence_length)
    if labels_weight is not None:
        labels_weight = torch.LongTensor(labels_weight)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, _, code_set, descs = tup
        data, target = torch.LongTensor(data), torch.FloatTensor(target)
        unseen_code_inds = unseen_code_inds.difference(code_set)
        if gpu:
            data = data.cuda()
            target = target.cuda()
            if labels_weight is not None:
                labels_weight = labels_weight.cuda()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        if args.model in ['bert', 'biobert', 'bert-tiny']:
            token_type_ids = (data > 0).long() * 0
            attention_mask = (data > 0).long()
            position_ids = torch.arange(data.size(1)).expand(data.size(0), data.size(1))
            if gpu:
                position_ids = position_ids.cuda()
            position_ids = position_ids * (data > 0).long()
        else:
            attention_mask = (data > 0).long()
            token_type_ids = None
            position_ids = None

        if args.model in BERT_MODEL_LIST:
            output, loss = model(input_ids=data, \
                                 token_type_ids=token_type_ids, \
                                 attention_mask=attention_mask, \
                                 position_ids=position_ids, \
                                 labels=target, \
                                 desc_data=desc_data, \
                                 pos_labels=labels_weight, \
                                )
        else:
            output, loss, _ = model(data, target, desc_data=desc_data)

        loss.backward()
        if args.model in BERT_MODEL_LIST:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.model in BERT_MODEL_LIST:
            scheduler.step()
        model.zero_grad()

        losses.append(loss.item())

        if not quiet and batch_idx % print_every == 0:
            #print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses, unseen_code_inds


def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    #wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    #replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0


def test(args, model, Y, epoch, data_path, fold, gpu, version, code_inds, dicts, samples, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    #initialize stuff for saving attention samples
    if samples:
        tp_file = open('%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open('%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        window_size = model.conv.weight.data.size()[2]

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    desc_embed = model.lmbda > 0
    if desc_embed and len(code_inds) > 0:
        unseen_code_vecs(model, code_inds, dicts, gpu)

    if args.model == 'bert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-base-uncased-vocab.txt', do_lower_case=True)
    elif args.model == 'biobert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=False)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/vocab.txt', do_lower_case=False)
    elif args.model == 'bert-tiny':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)
    else:
        bert_tokenizer = None

    model.eval()
    gen = datasets.data_generator(filename, dicts, 1, num_labels, version=version, desc_embed=desc_embed, bert_tokenizer=bert_tokenizer, test=True, max_seq_length=args.max_sequence_length)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, hadm_ids, _, descs = tup
        data, target = torch.LongTensor(data), torch.FloatTensor(target)
        if gpu:
            data = data.cuda()
            target = target.cuda()

        if desc_embed:
            desc_data = descs
        else:
            desc_data = None

        if args.model in ['bert', 'biobert', 'bert-tiny']:
            token_type_ids = (data > 0).long() * 0
            attention_mask = (data > 0).long()
            position_ids = torch.arange(data.size(1)).expand(data.size(0), data.size(1))
            if gpu:
                position_ids = position_ids.cuda()
            position_ids = position_ids * (data > 0).long()
        else:
            attention_mask = (data > 0).long()
            token_type_ids = None
            position_ids = None

        if args.model in BERT_MODEL_LIST:
            with torch.no_grad():
                output, loss = model(input_ids=data, \
                                     token_type_ids=token_type_ids, \
                                     attention_mask=attention_mask, \
                                     position_ids=position_ids, \
                                     labels=target, \
                                     desc_data=desc_data, \
                                     pos_labels=None, \
                                    )
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()
        else:
            with torch.no_grad():
                output, loss, alpha = model(data, target, desc_data=desc_data, get_attention=get_attn)


            #get an attention sample for 2% of batches
            get_attn = samples and (np.random.rand() < 0.02 or (fold == 'test' and testing))

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            if get_attn and samples:
                interpret.save_samples(data, output, target_data, alpha, window_size, epoch, tp_file, fp_file, dicts=dicts)

        losses.append(loss.item())
        target_data = target.data.cpu().numpy()

        #save predictions, target, hadm ids
        yhat_raw.append(output)
        output = np.round(output)
        y.append(target_data)
        yhat.append(output)
        hids.extend(hadm_ids)

    # close files if needed
    if samples:
        tp_file.close()
        fp_file.close()

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 50 else [8,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics


def pretrain(args, data_path):
    if args.model == 'bert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-base-uncased-vocab.txt', do_lower_case=True)
    elif args.model == 'biobert':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=False)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/vocab.txt', do_lower_case=False)
    elif args.model == 'bert-tiny':
        if args.redefined_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert-tiny-uncased-vocab.txt', do_lower_case=True)

    if args.model == 'bert':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-base-uncased-config.json')
        if args.Y == 'full':
            config.Y = 8921
        else:
            config.Y = int(args.Y)
        config.gpu = args.gpu
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model

        if args.from_scratch:
            model = BertForMaskedLM(config=config)
        else:
            model = BertForMaskedLM.from_pretrained('./pretrained_weights/bert-base-uncased-pytorch_model.bin', config=config)
    elif args.model == 'biobert':
        config = BertConfig.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/bert_config.json')
        if args.Y == 'full':
            config.Y = 8921
        else:
            config.Y = int(args.Y)
        config.gpu = args.gpu
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model
        if args.from_scratch:
            bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=False)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/vocab.txt', do_lower_case=False)
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.model = args.model
        if args.from_scratch:
            model = BertForMaskedLM(config=config)
        else:
            model = BertForMaskedLM.from_pretrained('./pretrained_weights/biobert_pretrain_output_all_notes_150000/pytorch_model.bin', config=config)
    elif args.model == 'bert-tiny':
        config = BertConfig.from_pretrained('./pretrained_weights/bert-tiny-uncased-config.json')
        if args.Y == 'full':
            config.Y = 8921
        else:
            config.Y = int(args.Y)
        config.gpu = args.gpu
        config.redefined_vocab_size = len(bert_tokenizer)
        config.redefined_max_position_embeddings = MAX_LENGTH
        config.last_module = args.last_module
        config.model = args.model
        if args.from_scratch:
            model = BertForMaskedLM(config=config)
        else:
            model = BertForMaskedLM.from_pretrained('./pretrained_weights/bert-tiny-uncased-pytorch_model.bin', config=config)

    if args.gpu:
        model.cuda()


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    pretrain_optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=args.weight_decay, lr=args.lr)
    length = datasets.data_length(args.data_path, args.version)
    t_total = length // args.pretrain_batch_size * args.pretrain_epochs
    pretrain_scheduler = get_linear_schedule_with_warmup(pretrain_optimizer, \
                                                         num_warmup_steps=args.warmup_steps, \
                                                         num_training_steps=t_total, \
                                                        )

    print_every = 25

    model.train()
    model.zero_grad()
    train_dataset = datasets.pretrain_data_generator(args, data_path, args.pretrain_batch_size, version=args.version, bert_tokenizer=bert_tokenizer, max_seq_length=args.max_sequence_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.pretrain_batch_size)

    for epoch in range(args.pretrain_epochs):
        losses = []
        for batch_idx, data in tqdm(enumerate(train_dataloader)):
            inputs, labels = random_mask_tokens(args, data, bert_tokenizer)
            if args.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            token_type_ids = (inputs > 0).long() * 0
            attention_mask = (inputs > 0).long()
            position_ids = torch.arange(inputs.size(1)).expand(inputs.size(0), inputs.size(1))
            if args.gpu:
                position_ids = position_ids.cuda()
            position_ids = position_ids * (inputs > 0).long()

            outputs = model(input_ids=inputs, \
                            token_type_ids=token_type_ids, \
                            attention_mask=attention_mask, \
                            position_ids=position_ids, \
                            masked_lm_labels=labels, \
                           )
            loss = outputs[0]
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            model.zero_grad()

            if batch_idx % print_every == 0:
                # print the average loss of the last 10 batches
                print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                    epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))

        loss = sum(losses) / len(losses)
        print('Epoch %d: %.4f' % (epoch, loss))

    model.save_pretrained(args.pretrain_ckpt_dir)
    print('Save pretrained model --> %s' % (args.pretrain_ckpt_dir))


def random_mask_tokens(args, inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)

    labels = inputs.clone()

    input_mask = (~inputs.eq(0)).to(torch.float)

    masking_prob = 0.15
    # Consider padding
    masked_indices = torch.bernoulli(input_mask * masking_prob).bool()
    labels[~masked_indices] = -1

    indices_replaced = torch.bernoulli(input_mask * 0.8).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(input_mask * 0.5).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("model", type=str, \
                        choices=["cnn_vanilla", "rnn", \
                                 "conv_attn", "multi_conv_attn", "logreg", "saved", "bert", "biobert", \
                                 "bert-caml", "bert-small-caml",
                                 "bert-tiny-caml", 'bert-tiny', 'bert-tiny-parallel-caml', \
                                ], \
                                help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)", dest='cell_type', default='gru')
    parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
                        help="optional flag for rnn to use a bidirectional model")
    parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=4,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--pool", choices=['max', 'avg'], required=False, dest="pool", help="which type of pooling to do (logreg model only)")
    parser.add_argument("--code-emb", type=str, required=False, dest="code_emb",
                        help="point to code embeddings to use for parameter initialization, if applicable")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=8,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--lmbda", type=float, required=False, dest="lmbda", default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3', required=False,
                        help="version of MIMIC in use (default: mimic3)")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--public-model", dest="public_model", action="store_const", required=False, const=True,
                        help="optional flag for testing pre-trained models from the public github")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--redefined_tokenizer', action='store_true')
    # parser.add_argument('--tokenizer_path', type=str, default='./tokenizers/bio-mimic3-6000-limit-10000-vocab.txt')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizers/bert-tiny-mimic3-100-limit-100000-vocab.txt')
    parser.add_argument('--last_module', type=str, default='basic')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument("--pretrain-batch-size", type=int, default=2)
    # parser.add_argument('--pretrain_datafile', type=str, default='./mimicdata/mimic3/pretrain_bert_512')
    parser.add_argument('--pretrain_datafile', type=str, default='./mimicdata/mimic3/pretrain_bert_tiny_512')
    # parser.add_argument('--pretrain_datafile', type=str, default='./mimicdata/mimic3/pretrain_bert_tiny_2500')
    parser.add_argument('--pretrain_epochs', type=int, default=3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000))
    parser.add_argument('--max_sequence_length', type=int, default=None)
    parser.add_argument('--cuda_device_no', type=int, default=None)
    parser.add_argument('--bert_parallel_count', type=int, default=None)
    parser.add_argument('--bert_parallel_final_layer', type=str, choices=['sum', 'cat'], default='sum')
    args = parser.parse_args()
    print('args', args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.cuda_device_no is not None:
        torch.cuda.set_device(args.cuda_device_no)

    command = ' '.join(['python'] + sys.argv)
    args.command = command

    if args.pretrain:
        args.pretrain_ckpt_dir = os.path.join(MODEL_DIR, 'pretrain')
        os.makedirs(args.pretrain_ckpt_dir, exist_ok=True)
    main(args)
