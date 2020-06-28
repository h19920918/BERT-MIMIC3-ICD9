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

# download google bert-base
if not os.path.isfile('pretrained_weights/bert-base-uncased-config.json'):
    url = 'https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/config.json'
    r = requests.get(url)
    with open('pretrained_weights/bert-base-uncased-config.json', 'wb') as f:
        f.write(r.content)
if not os.path.isfile('pretrained_weights/bert-base-uncased-pytorch_model.bin'):
    url = 'https://cdn.huggingface.co/google/bert_uncased_L-12_H-768_A-12/pytorch_model.bin'
    r = requests.get(url)
    with open('pretrained_weights/bert-base-uncased-pytorch_model.bin', 'wb') as f:
        f.write(r.content)
if not os.path.isfile('pretrained_weights/bert-base-uncased-vocab.txt'):
    url = 'https://cdn.huggingface.co/google/bert_uncased_L-12_H-768_A-12/vocab.txt'
    r = requests.get(url)
    with open('pretrained_weights/bert-base-uncased-vocab.txt', 'wb') as f:
        f.write(r.content)