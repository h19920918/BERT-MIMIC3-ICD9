import sys
sys.path.append('../')
import datasets
import log_reg
from dataproc import extract_wvs
from dataproc import get_discharge_summaries
from dataproc import concat_and_split
from dataproc import build_vocab
from dataproc import vocab_index_descriptions
from dataproc import word_embeddings
from constants import MIMIC_3_DIR, DATA_DIR

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
import csv
import math
import operator
import os
import shutil
import requests

# download huggingface pretrained weights
if not os.path.exists('pretrained_weights'):
    os.mkdir('pretrained_weights')
    print('mkdir', 'pretrained_weights')

# download google bert-tiny
if not os.path.isfile('pretrained_weights/bert-tiny-uncased-config.json'):
    url = 'https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-2_H-128_A-2/config.json'
    r = requests.get(url)
    with open('pretrained_weights/bert-tiny-uncased-config.json', 'wb') as f:
        f.write(r.content)
if not os.path.isfile('pretrained_weights/bert-tiny-uncased-pytorch_model.bin'):
    url = 'https://cdn.huggingface.co/google/bert_uncased_L-2_H-128_A-2/pytorch_model.bin'
    r = requests.get(url)
    with open('pretrained_weights/bert-tiny-uncased-pytorch_model.bin', 'wb') as f:
        f.write(r.content)
if not os.path.isfile('pretrained_weights/bert-tiny-uncased-vocab.txt'):
    url = 'https://cdn.huggingface.co/google/bert_uncased_L-2_H-128_A-2/vocab.txt'
    r = requests.get(url)
    with open('pretrained_weights/bert-tiny-uncased-vocab.txt', 'wb') as f:
        f.write(r.content)

# copy mimic3 data for https://github.com/jamesmullenbach/caml-mimic#data-processing
if len(sys.argv) >= 2:
    mimic3_original_path = sys.argv[1]
    if os.path.exists(mimic3_original_path):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
            print('mkdir', DATA_DIR)
        if not os.path.exists(MIMIC_3_DIR):
            os.mkdir(MIMIC_3_DIR)
            print('mkdir', MIMIC_3_DIR)

        if not os.path.isfile(DATA_DIR + '/D_ICD_DIAGNOSES.csv'):
            shutil.copy(mimic3_original_path + '/D_ICD_DIAGNOSES.csv', DATA_DIR + '/D_ICD_DIAGNOSES.csv')
            print('copy', mimic3_original_path + '/D_ICD_DIAGNOSES.csv', DATA_DIR + '/D_ICD_DIAGNOSES.csv')
            
        if not os.path.isfile(DATA_DIR + '/D_ICD_PROCEDURES.csv'):
            shutil.copy(mimic3_original_path + '/D_ICD_PROCEDURES.csv', DATA_DIR + '/D_ICD_PROCEDURES.csv')
            print('copy', mimic3_original_path + '/D_ICD_PROCEDURES.csv', DATA_DIR + '/D_ICD_PROCEDURES.csv')
            
        if not os.path.isfile(MIMIC_3_DIR + '/NOTEEVENTS.csv'):
            shutil.copy(mimic3_original_path + '/NOTEEVENTS.csv', MIMIC_3_DIR + '/NOTEEVENTS.csv')
            print('copy', mimic3_original_path + '/NOTEEVENTS.csv', MIMIC_3_DIR + '/NOTEEVENTS.csv')
            
        if not os.path.isfile(MIMIC_3_DIR + '/DIAGNOSES_ICD.csv'):
            shutil.copy(mimic3_original_path + '/DIAGNOSES_ICD.csv', MIMIC_3_DIR + '/DIAGNOSES_ICD.csv')
            print('copy', mimic3_original_path + '/DIAGNOSES_ICD.csv', MIMIC_3_DIR + '/DIAGNOSES_ICD.csv')
            
        if not os.path.isfile(MIMIC_3_DIR + '/PROCEDURES_ICD.csv'):
            shutil.copy(mimic3_original_path + '/PROCEDURES_ICD.csv', MIMIC_3_DIR + '/PROCEDURES_ICD.csv')
            print('copy', mimic3_original_path + '/PROCEDURES_ICD.csv', MIMIC_3_DIR + '/PROCEDURES_ICD.csv')
            
        if not os.path.isfile(MIMIC_3_DIR + '/PROCEDURES_ICD.csv'):
            shutil.copy(mimic3_original_path + '/PROCEDURES_ICD.csv', MIMIC_3_DIR + '/PROCEDURES_ICD.csv')
            print('copy', mimic3_original_path + '/PROCEDURES_ICD.csv', MIMIC_3_DIR + '/PROCEDURES_ICD.csv')

biobert = True
Y = 'full' # use all available labels in the dataset for prediction
notes_file = '%s/NOTEEVENTS.csv' % MIMIC_3_DIR # raw note events downloaded from MIMIC-III
vocab_size = 'full' # don't limit the vocab size to a specific number
vocab_min = 3 # discard tokens appearing in fewer than this many documents

dfproc = pd.read_csv('%s/PROCEDURES_ICD.csv' % MIMIC_3_DIR)
dfdiag = pd.read_csv('%s/DIAGNOSES_ICD.csv' % MIMIC_3_DIR)

dfdiag['absolute_code'] = dfdiag.apply(lambda row: str(datasets.reformat(str(row[4]), True)), axis=1)
dfproc['absolute_code'] = dfproc.apply(lambda row: str(datasets.reformat(str(row[4]), False)), axis=1)

dfcodes = pd.concat([dfdiag, dfproc])

dfcodes.to_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR, index=False,
               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])

# In the full dataset (not just discharge summaries)
df = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR, dtype={"ICD9_CODE": str})
print(len(df['ICD9_CODE'].unique()))

# This reads all notes, selects only the discharge summaries, and tokenizes them, returning the output filename
disch_full_file = get_discharge_summaries.write_discharge_summaries(out_file="%s/disch_full.csv" % MIMIC_3_DIR, biobert=biobert)

df = pd.read_csv('%s/disch_full.csv' % MIMIC_3_DIR)

# How many admissions?
print(len(df['HADM_ID'].unique()))

# Tokens and types
types = set()
num_tok = 0
for row in df.itertuples():
    for w in row[4].split():
        types.add(w)
        num_tok += 1

print("Num types", len(types))
print("Num tokens", str(num_tok))

# Let's sort by SUBJECT_ID and HADM_ID to make a correspondence with the MIMIC-3 label file
df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])

# Sort the label file by the same
dfl = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_3_DIR)
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])

print(len(df['HADM_ID'].unique()), len(dfl['HADM_ID'].unique()))

# Let's filter out these HADM_ID's
hadm_ids = set(df['HADM_ID'])
with open('%s/ALL_CODES.csv' % MIMIC_3_DIR, 'r') as lf:
    with open('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, 'w') as of:
        w = csv.writer(of)
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
        r = csv.reader(lf)
        #header
        next(r)
        for i,row in enumerate(r):
            hadm_id = int(row[2])
            #print(hadm_id)
            #break
            if hadm_id in hadm_ids:
                w.writerow(row[1:3] + [row[-1], '', ''])

dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index_col=None)

print(len(dfl['HADM_ID'].unique()))

# we still need to sort it by HADM_ID
dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfl.to_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index=False)

# Now let's append each instance with all of its codes
# this is pretty non-trivial so let's use this script I wrote, which requires the notes to be written to file
sorted_file = '%s/disch_full.csv' % MIMIC_3_DIR
df.to_csv(sorted_file, index=False)

labeled = concat_and_split.concat_data('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, sorted_file)

# name of the file we just made
print(labeled)


dfnl = pd.read_csv(labeled)
# Tokens and types
types = set()
num_tok = 0
for row in dfnl.itertuples():
    for w in row[3].split():
        types.add(w)
        num_tok += 1

print("num types", len(types), "num tokens", num_tok)
print(len(dfnl['HADM_ID'].unique()))

fname = '%s/notes_labeled.csv' % MIMIC_3_DIR
base_name = "%s/disch" % MIMIC_3_DIR #for output
tr, dv, te = concat_and_split.split_data(fname, base_name=base_name)

vocab_min = 3
vname = '%s/vocab.csv' % MIMIC_3_DIR
build_vocab.build_vocab(vocab_min, tr, vname)

for splt in ['train', 'dev', 'test']:
    filename = '%s/disch_%s_split.csv' % (MIMIC_3_DIR, splt)
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_full.csv' % (MIMIC_3_DIR, splt), index=False)

w2v_file = word_embeddings.word_embeddings('full', '%s/disch_full.csv' % MIMIC_3_DIR, 100, 0, 5)

extract_wvs.gensim_to_embeddings('%s/processed_full.w2v' % MIMIC_3_DIR, '%s/vocab.csv' % MIMIC_3_DIR, Y)

vocab_index_descriptions.vocab_index_descriptions('%s/vocab.csv' % MIMIC_3_DIR,
                                                  '%s/description_vectors.vocab' % MIMIC_3_DIR)

Y = 50

# first calculate the top k
counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled.csv' % MIMIC_3_DIR)
for row in dfnl.itertuples():
    for label in str(row[4]).split(';'):
        counts[label] += 1

codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

codes_50 = [code[0] for code in codes_50[:Y]]

print(codes_50)

with open('%s/TOP_%s_CODES.csv' % (MIMIC_3_DIR, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])

for splt in ['train', 'dev', 'test']:
    print(splt)
    hadm_ids = set()
    with open('%s/%s_50_hadm_ids.csv' % (MIMIC_3_DIR, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())
    with open('%s/notes_labeled.csv' % MIMIC_3_DIR, 'r') as f:
        with open('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                hadm_id = row[1]
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[3]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:3] + [';'.join(filtered_codes)])
                    i += 1

for splt in ['train', 'dev', 'test']:
    filename = '%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y))
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), index=False)
