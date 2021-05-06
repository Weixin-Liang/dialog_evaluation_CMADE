# -*- coding: utf-8 -*-
# file: data_utils.py

import os
import pickle
import numpy as np
import math 
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import Counter 
import random


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset_window(Dataset):

    def __init__(self, dataset_list=None, tokenizer=None, balanced_upsampling_flag=True):
        
        self.balanced_upsampling_flag = balanced_upsampling_flag

        if dataset_list is None and tokenizer is None:
            self.data = []
            return 

        non_terminate_windows, terminate_windows = dataset_list

        stop_data = []
        nonstop_data = []
        

        def tokenize_helper(dataset_list, polarity, target_list):
            truncated_count = 0
            for i in range(len(dataset_list)):

                text_left_join = '[CLS] '
                bert_segments_ids = []

                text_left_list = []
                text_left_ids_list = []
                text_right_list = []
                text_right_ids_list = []
                for j in range(len(dataset_list[i]['text'])):
                    
                    text_left = dataset_list[i]['last_response'][j].lower().strip()
                    text_right = dataset_list[i]['text'][j].lower().strip()
                    left_raw_indices = tokenizer.text_to_sequence(text_left, truncating='pre')
                    right_raw_indices = tokenizer.text_to_sequence(text_right, truncating='pre')
                    text_left_join += text_left + " [SEP] " + text_right + " [SEP] "
                
                text_bert_indices = tokenizer.text_to_sequence(text_left_join, truncating='pre')    

                this_bert_segments_ids = ([0] * (np.sum(left_raw_indices != 0) + 1)  + [1] * (np.sum(right_raw_indices != 0) + 1))
                bert_segments_ids += this_bert_segments_ids

                if len(bert_segments_ids) > tokenizer.max_seq_len:
                    truncated_count += 1
                    print("truncated_count", truncated_count, "len", len(bert_segments_ids), "text_left_join", text_left_join)

                bert_segments_ids = np.asarray(bert_segments_ids)
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len, truncating='pre')


                data = {
                    'text_bert_indices': text_bert_indices, 
                    'bert_segments_ids': bert_segments_ids, 
                    'polarity': polarity,
                    "text_left_join": text_left_join, 
                    'conversation_id': dataset_list[i]['conversation_id'],
                    'turn_j': str(dataset_list[i]['turn_j']), 
                }

                target_list.append(data)


        print("non_terminate_windows", len(non_terminate_windows))
        tokenize_helper(dataset_list=non_terminate_windows, polarity=0, target_list=nonstop_data)
        
        print("terminate_windows", len(terminate_windows))
        tokenize_helper(dataset_list=terminate_windows, polarity=1, target_list=stop_data)

        stop_count = len(stop_data)
        non_stop_count = len(nonstop_data)

        if self.balanced_upsampling_flag: 
            max_len = max(len(nonstop_data), len(stop_data))
            up_sampled_stop_data = math.ceil(max_len/len(stop_data)) * stop_data
            up_sampled_stop_data = up_sampled_stop_data[:max_len]
        
            up_sampled_non_stop_data = math.ceil(max_len/len(nonstop_data)) * nonstop_data
            up_sampled_non_stop_data = up_sampled_non_stop_data[:max_len]

            self.data = up_sampled_stop_data + up_sampled_non_stop_data
        else:
            self.data = stop_data + nonstop_data 
        print("dataset summary:", "stop_count", stop_count, "non_stop_count", non_stop_count, "dataset len", len(self.data) )


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass