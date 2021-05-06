# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from collections import Counter 

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


class ABSADataset_list(Dataset):
    def __init__(self, dataset_list, tokenizer):

        dial_num = len(dataset_list)

        all_data = []
        for i in range(0, dial_num):
            polarity = dataset_list[i]['rating']-1
            assert type(polarity) is int and polarity>=0 and polarity<=4, dataset_list[i]

            if polarity > 1:
                polarity = 1
            else:
                polarity = 0

            text_bert_indices_list = []
            bert_segments_ids_list = []

            for j in range(len(dataset_list[i]['text'])):
                if type(dataset_list[i]['text'][j]) is not str or \
                j>=len(dataset_list[i]['response']) or \
                type(dataset_list[i]['response'][j]) is not str or \
                dataset_list[i]['text'][j]=="" or \
                dataset_list[i]['response'][j]=="":
                    break
                
                text_left = dataset_list[i]['text'][j].lower().strip()

                left_raw_indices = tokenizer.text_to_sequence(text_left)
                text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + ' [SEP]')
                bert_segments_ids = np.asarray([0] * (np.sum(left_raw_indices != 0) + 2)       )
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)


                text_bert_indices_list.append(text_bert_indices)
                bert_segments_ids_list.append(bert_segments_ids)
                
            
            data = {
                'text_bert_indices': text_bert_indices_list, 
                'bert_segments_ids': bert_segments_ids_list, 
                'polarity': polarity,
                'data_id': len(all_data), 
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def generate_colosal_training_data(tokenizer):
    def load_func(line):
        return {'src': src, 'target': target} 

    def batchify(batch):
        return {'src': batch_src, 'target': batch_target} 

    dataset = ListDataset('list.txt', load_func) 
    dataset = DataLoader(dataset=dataset, batch_size=50, num_workers=8, collate_fn=batchify)

    return None


class ABSADataset(Dataset): # save 

    def __init__(self, dataset_list=None, tokenizer=None):

        if dataset_list is None and tokenizer is None:
            self.data = []
            return 

        dial_num = len(dataset_list)

        all_data = []
        truncated_count = 0
        for i in range(0, dial_num):
            rating = dataset_list[i]['rating']-1
            polarity = rating
            assert type(polarity) is int and polarity>=0 and polarity<=4, dataset_list[i]

            if polarity > 1:
                polarity = 1
            else:
                polarity = 0

            text_bert_indices_list = ['[CLS] ']
            bert_segments_ids_list = [ [0] ]
            text_left_join = '[CLS] '
            bert_segments_ids = []

            # no CLS
            text_left_list = []
            text_left_ids_list = []
            text_right_list = []
            text_right_ids_list = []
            

            for j in range(len(dataset_list[i]['text'])):
                if type(dataset_list[i]['text'][j]) is not str or \
                j>=len(dataset_list[i]['response']) or \
                type(dataset_list[i]['response'][j]) is not str or \
                dataset_list[i]['text'][j]=="" or \
                dataset_list[i]['response'][j]=="":
                    break
                text_left = dataset_list[i]['text'][j].lower().strip()
                text_right = dataset_list[i]['response'][j].lower().strip()
                left_raw_indices = tokenizer.text_to_sequence(text_left)
                right_raw_indices = tokenizer.text_to_sequence(text_right)
                text_left_join += text_left + " [SEP] " + text_right + " [SEP] "

                if j == 0:
                    text_bert_indices = (text_left + " [SEP] " + text_right + " [SEP] ")
                    this_bert_segments_ids = ([0] * (np.sum(left_raw_indices != 0) + 1)  + [1] * (np.sum(right_raw_indices != 0) + 1))
                else:
                    text_bert_indices = (text_left + " [SEP] " + text_right + " [SEP] ")
                    this_bert_segments_ids = ([0] * (np.sum(left_raw_indices != 0) + 1)  + [1] * (np.sum(right_raw_indices != 0) + 1))
                
                bert_segments_ids += this_bert_segments_ids

                # for turn-level obfuscation
                text_bert_indices_list.append(text_bert_indices)
                bert_segments_ids_list.append(bert_segments_ids)

                # for utterance-level obfuscation
                text_left_list.append(text_left+ " [SEP] ")
                text_right_list.append(text_right + " [SEP] ")

                text_left_ids_list.append([0] * (np.sum(left_raw_indices != 0) + 1))
                text_right_ids_list.append([1] * (np.sum(right_raw_indices != 0) + 1))

            
            text_bert_indices = tokenizer.text_to_sequence(text_left_join)    

            if len(bert_segments_ids) > tokenizer.max_seq_len:
                # log warning? 
                truncated_count += 1
                print("truncated_count", truncated_count, "len", len(bert_segments_ids))

            bert_segments_ids = np.asarray(bert_segments_ids)
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)


            data = {
                'text_bert_indices': [text_bert_indices], 
                'bert_segments_ids': [bert_segments_ids], 
                
                # turn level
                'text_bert_indices_list': text_bert_indices_list, 
                'bert_segments_ids_list': bert_segments_ids_list,

                # utterance level
                'text_left_list': text_left_list,
                'text_right_list': text_right_list,
                'text_left_ids_list': text_left_ids_list,
                'text_right_ids_list': text_right_ids_list, 
                'polarity': polarity,
                "rating": str(rating), 
                "rating_num": int(rating), 
                'data_id': len(all_data), 
                'conversation_id': dataset_list[i]['conversation_id'], 
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADataset_save(Dataset): 

    def __init__(self, dataset_list, tokenizer):

        dial_num = len(dataset_list)

        all_data = []
        for i in range(0, dial_num):
            polarity = dataset_list[i]['rating']-1
            assert type(polarity) is int and polarity>=0 and polarity<=4, dataset_list[i]
            

            if polarity > 1:
                polarity = 1
            else:
                polarity = 0

            text_left_join = '[CLS] '
            text_left_join_len = 1

            for j in range(len(dataset_list[i]['text'])):
                if type(dataset_list[i]['text'][j]) is not str or \
                j>=len(dataset_list[i]['response']) or \
                type(dataset_list[i]['response'][j]) is not str or \
                dataset_list[i]['text'][j]=="" or \
                dataset_list[i]['response'][j]=="":
                    break
                text_left = dataset_list[i]['text'][j].lower().strip()
                left_raw_indices = tokenizer.text_to_sequence(text_left)
                text_left_join_len += (np.sum(left_raw_indices != 0) + 1)
                text_left_join = text_left_join + text_left + " [SEP] "
                """
                text_left = dataset_list[i]['text'][j].lower().strip()
                text_right = dataset_list[i]['response'][j].lower().strip()

                left_raw_indices = tokenizer.text_to_sequence(text_left)
                # right_raw_indices = tokenizer.text_to_sequence(text_right)
                text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + ' [SEP]')
                bert_segments_ids = np.asarray([0] * (np.sum(left_raw_indices != 0) + 2) )# + [1] * (np.sum(right_raw_indices != 0) + 1))
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)


                text_bert_indices_list.append(text_bert_indices)
                bert_segments_ids_list.append(bert_segments_ids)
                """
            
            text_bert_indices = tokenizer.text_to_sequence(text_left_join)    
            bert_segments_ids = np.asarray([0] * text_left_join_len )
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            data = {
                'text_bert_indices': [text_bert_indices], 
                'bert_segments_ids': [bert_segments_ids], 
                'polarity': polarity,
                'data_id': len(all_data), 
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class AnnotatedPairTieDataset(ABSADataset):

    def __init__(self, rawtest, testset, annotaed_pairs):        
        self.data = []

        def match_in_test(conversation_id):
            for i in range(len(rawtest)):
                this_conversation_id = rawtest[i]['conversation_id']
                if this_conversation_id == conversation_id:
                    return i
            
            raise NotImplementedError("Could not match in the test dataset, conversation_id: ",conversation_id)
        
        agree_count = 0
        distance_counter = Counter()
        for new_row in annotaed_pairs:
            dial1_idx = match_in_test(new_row['dial1_id'])
            dial2_idx = match_in_test(new_row['dial2_id'])
            one_pair_data_point = {
                'dial1': testset[dial1_idx], 
                'dial2': testset[dial2_idx], 
            }
            self.data.append(one_pair_data_point)
            raw_rating1 = rawtest[dial1_idx]['rating']
            raw_rating2 = rawtest[dial2_idx]['rating']
            distance = abs(raw_rating1-raw_rating2)
            distance_counter[distance] += 1
            print(raw_rating1, raw_rating2, "tie")



    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)





class AnnotatedPairTestDataset(ABSADataset):

    def __init__(self, rawtest, testset, annotaed_pairs):
        # super.__init__(testset)
        
        self.data = []

        def match_in_test(conversation_id):
            for i in range(len(rawtest)):
                # print("rawtest[i]", rawtest[i])
                this_conversation_id = rawtest[i]['conversation_id']
                if this_conversation_id == conversation_id:
                    return i
            
            raise NotImplementedError("Could not match in the test dataset, conversation_id: ",conversation_id)
        
        agree_count = 0
        ambiguous_count = 0
        agree_distance_counter = Counter()
        disagree_distance_counter = Counter()
        for new_row in annotaed_pairs:
            dial1_idx = match_in_test(new_row['dial1_id'])
            dial2_idx = match_in_test(new_row['dial2_id'])
            compare_res = new_row['compare_res']
            assert(new_row['compare_res'] == 1 or new_row['compare_res'] == 2)
            one_pair_data_point = {
                'dial1': testset[dial1_idx], 
                'dial2': testset[dial2_idx], 
                'compare_res': compare_res, 
            }
            self.data.append(one_pair_data_point)
            raw_rating1 = rawtest[dial1_idx]['rating']
            raw_rating2 = rawtest[dial2_idx]['rating']
            raw_rating_cmp = int(raw_rating1<raw_rating2)
            annotated_cmp = compare_res-1
            if raw_rating1==raw_rating2:
                ambiguous_count+=1
            elif raw_rating_cmp == annotated_cmp:
                agree_count+=1
                distance = abs(raw_rating1-raw_rating2)
                agree_distance_counter[distance] += 1
            else:
                # disagree
                distance = abs(raw_rating1-raw_rating2)
                disagree_distance_counter[distance] += 1
            print(raw_rating1, raw_rating2, "annotated_cmp", annotated_cmp)

        # inter-agreement measurement
        print("agree_distance_counter", agree_distance_counter)
        print("disagree_distance_counter", disagree_distance_counter)
        print( "agree_count", agree_count, "ambiguous_count: ", ambiguous_count)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass