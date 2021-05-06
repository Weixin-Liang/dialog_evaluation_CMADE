

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy  as np

from pytorch_pretrained_bert import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data_utils import Tokenizer4Bert, ABSADataset, AnnotatedPairTestDataset, AnnotatedPairTieDataset, pad_and_truncate
import pickle 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

from dataio import load_annotated_pairs, load_dataset, balanced_dataset_load, load_annotated_tie_pairs #load_data, split_dataset, 
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve, classification_report, confusion_matrix
from scipy.stats import spearmanr, pearsonr


from algorithm_utils import knn_smooth_scores, do_knn_shapley

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        self.tokenizer = tokenizer

        pkl_path = "./datacache_bertspc.pkl"
        regenerate_data_flag = True  #True
        if not regenerate_data_flag and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as pkl_file: 
                data_dict = pickle.load(pkl_file)
                self.tmp_all = data_dict['self.tmp_all']
                self.tmp_all_encode = data_dict['self.tmp_all_encode']
                self.train_all = data_dict['self.train_all']
                self.trainset = data_dict['self.trainset']
                self.annotaed_pair_test_dataset = data_dict['self.annotaed_pair_test_dataset']
                self.annotaed_pair_val_dataset  = data_dict['self.annotaed_pair_val_dataset']
                self.used_dialog_ids = data_dict['self.used_dialog_ids']
                self.annotaed_pair_tie_dataset = data_dict['self.annotaed_pair_tie_dataset']
        else:

            tmp_all = load_dataset(data_set_csv_file_name='./paircourpus.csv', used_dialog_ids=set(), early_truncate=800)
            tmp_all_encode = ABSADataset(tmp_all, tokenizer)
            self.tmp_all = tmp_all
            self.tmp_all_encode = tmp_all_encode
            
            annotaed_pairs, used_dialog_ids = load_annotated_pairs(meta_annotation_csv_path="./cleanedpairs403.csv") # "./morepairs459.csv
            # used_dialog_ids: indicates the dialog that should not be included inside the training set. 
            
            dev_annotaed_pairs =  annotaed_pairs[-200:] 
            test_annotaed_pairs = annotaed_pairs[:-200] 
            self.annotaed_pair_val_dataset = AnnotatedPairTestDataset(self.tmp_all, self.tmp_all_encode, dev_annotaed_pairs)
            self.annotaed_pair_test_dataset = AnnotatedPairTestDataset(self.tmp_all, self.tmp_all_encode, test_annotaed_pairs)

            tie_annotaed_pairs, tie_used_dialog_ids = load_annotated_tie_pairs(meta_annotation_csv_path="./tiepairs166.csv") # "./morepairs459.csv
            self.annotaed_pair_tie_dataset = AnnotatedPairTieDataset(self.tmp_all, self.tmp_all_encode, tie_annotaed_pairs)
            
            self.used_dialog_ids = used_dialog_ids.union(tie_used_dialog_ids)
            self.train_all = load_dataset(data_set_csv_file_name='./save10clean.csv', used_dialog_ids=self.used_dialog_ids)
            self.trainset = ABSADataset(self.train_all, tokenizer)

            # for pairs. 
            datasets = balanced_dataset_load(data_set_csv_file_name='./save10clean.csv', used_dialog_ids=self.used_dialog_ids)
            

            with open(pkl_path, 'wb') as pkl_file:
                data_dict = {
                    "self.train_all": self.train_all, 
                    "self.trainset": self.trainset,
                    "self.tmp_all_encode": self.tmp_all_encode, 
                    "self.tmp_all": self.tmp_all,
                    "self.annotaed_pair_test_dataset": self.annotaed_pair_test_dataset, 
                    "self.annotaed_pair_val_dataset": self.annotaed_pair_val_dataset, 
                    'self.used_dialog_ids': self.used_dialog_ids, 
                    "self.annotaed_pair_tie_dataset": self.annotaed_pair_tie_dataset, 
                }
                pickle.dump(data_dict, pkl_file)
            
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            opt.batch_size *= torch.cuda.device_count()

        self.model = self.model.to(opt.device)


        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # helper for stage 1 
    def fake_a_batch(self, real_batch, trainset):
        # in-place
        fake_user_batch = {
            'text_bert_indices': [],
            'bert_segments_ids': [],
        }     
        fake_sys_batch = {
            'text_bert_indices': [],
            'bert_segments_ids': [],
        }        
        data_ids_list = real_batch['data_id'].numpy().tolist()

        for i, data_id in enumerate(data_ids_list):
            next_i = (i+1) % len(data_ids_list)
            #
            datapoint = trainset[data_id]
            
            # for user
            text_left_list = datapoint['text_left_list'].copy()
            text_left_ids_list = datapoint['text_left_ids_list'].copy()

            text_right_list = datapoint['text_right_list'].copy()
            text_right_ids_list = datapoint['text_right_ids_list'].copy()
            dial_len = len(text_left_list)

            # for system
            sys_text_left_list = datapoint['text_left_list'].copy()
            sys_text_left_ids_list = datapoint['text_left_ids_list'].copy()

            sys_text_right_list = datapoint['text_right_list'].copy()
            sys_text_right_ids_list = datapoint['text_right_ids_list'].copy()

            next_data_id = data_ids_list[next_i]
            next_datapoint = trainset[next_data_id]
            next_text_left_list = next_datapoint['text_left_list']
            next_text_left_ids_list = next_datapoint['text_left_ids_list']
            next_text_right_list = next_datapoint['text_right_list']
            next_text_right_ids_list = next_datapoint['text_right_ids_list']
            next_dial_len = len(next_text_left_list)

            src_swap_list = random.sample(range(0, dial_len), 2) # 0 for user, 1 for system. the idx in this dialog to be replaced
            next_swap_list = random.sample(range(0, next_dial_len), 2) # the idx in the next dialog serving as the replacement

            text_left_list[src_swap_list[0]] = next_text_left_list[next_swap_list[0]]
            text_left_ids_list[src_swap_list[0]]= next_text_left_ids_list[next_swap_list[0]] 
            
            sys_text_right_list[src_swap_list[1]] = next_text_right_list[next_swap_list[1]]
            sys_text_right_ids_list[src_swap_list[1]]= next_text_right_ids_list[next_swap_list[1]] 
            

            # user generate
            new_bert_segments_ids = [0]
            for l, r in zip(text_left_ids_list, text_right_ids_list):
                new_bert_segments_ids.extend(l)
                new_bert_segments_ids.extend(r)

            new_bert_segments_ids = np.asarray(new_bert_segments_ids)
            new_bert_segments_ids = pad_and_truncate(new_bert_segments_ids, self.tokenizer.max_seq_len)

            new_text_bert_indices = "[CLS] "
            for l, r in zip(text_left_list, text_right_list):
                new_text_bert_indices += (l+r)

            new_text_bert_indices = self.tokenizer.text_to_sequence(new_text_bert_indices) 

            new_text_bert_indices = torch.from_numpy(new_text_bert_indices)
            new_bert_segments_ids = torch.from_numpy(new_bert_segments_ids)
            

            fake_user_batch['text_bert_indices'].append(new_text_bert_indices)
            fake_user_batch['bert_segments_ids'].append(new_bert_segments_ids)

            # ------   fake a system now: system generate 
            sys_new_bert_segments_ids = [0]
            for l, r in zip(sys_text_left_ids_list, sys_text_right_ids_list):
                sys_new_bert_segments_ids.extend(l)
                sys_new_bert_segments_ids.extend(r)

            sys_new_bert_segments_ids = np.asarray(sys_new_bert_segments_ids)
            sys_new_bert_segments_ids = pad_and_truncate(sys_new_bert_segments_ids, self.tokenizer.max_seq_len)

            sys_new_text_bert_indices = "[CLS] "
            for l, r in zip(sys_text_left_list, sys_text_right_list):
                sys_new_text_bert_indices += (l+r)

            sys_new_text_bert_indices = self.tokenizer.text_to_sequence(sys_new_text_bert_indices) 

            sys_new_text_bert_indices = torch.from_numpy(sys_new_text_bert_indices)
            sys_new_bert_segments_ids = torch.from_numpy(sys_new_bert_segments_ids)

            fake_sys_batch['text_bert_indices'].append(sys_new_text_bert_indices)
            fake_sys_batch['bert_segments_ids'].append(sys_new_bert_segments_ids)

        
        fake_user_batch['text_bert_indices'] = torch.stack(fake_user_batch['text_bert_indices']).unsqueeze(0)
        fake_user_batch['bert_segments_ids'] = torch.stack(fake_user_batch['bert_segments_ids']).unsqueeze(0)

        fake_sys_batch['text_bert_indices'] = torch.stack(fake_sys_batch['text_bert_indices']).unsqueeze(0)
        fake_sys_batch['bert_segments_ids'] = torch.stack(fake_sys_batch['bert_segments_ids']).unsqueeze(0)

        return fake_user_batch, fake_sys_batch


    def _train_stage_3(self, criterion, optimizer, train_data_loader):
        #######################
        # Stage 3: Denoising with data Shapley &further fine-tuning 
        #######################
        max_val_acc = 0 
        max_val_f1  = 0 
        global_step = 0 
        path = None 

        # cmp_merge_sigmoid = nn.Sigmoid()
        bce_loss_fun = nn.BCEWithLogitsLoss().cuda() # torch.nn.BCELoss().cuda()

        pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   


        for epoch in range(5): #range(self.opt.num_epoch):
            
            # # reset the whole model 
            # self._reset_params()
            # self.bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)

            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            optimizer.zero_grad()
            prev_sample_batched = None
            for sample_batched in train_data_loader:
                if prev_sample_batched is None:
                    prev_sample_batched = sample_batched
                    continue
                
                global_step += 1
                # clear gradient accumulators


                # inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                inputs = [[x.to(self.opt.device) for x in sample_batched[col]]  for col in self.opt.inputs_cols]
                
                # generate a fake batch here 
                real_outputs, _ = self.model(inputs)
                real_outputs = real_outputs.squeeze(1)
                

                
                prev_inputs = [[x.to(self.opt.device) for x in prev_sample_batched[col]]  for col in self.opt.inputs_cols]
                prev_outputs, _ = self.model(prev_inputs)
                prev_outputs = prev_outputs.squeeze(1)
                
                
                prev_ratings = prev_sample_batched['rating_num']
                real_ratings = sample_batched['rating_num']
                targets = torch.le(prev_ratings, real_ratings).float()
                targets = targets.to(self.opt.device)

                # prev_ids_np = prev_sample_batched['data_id'].numpy()
                # real_ids_np = sample_batched['data_id'].numpy()
                # prev_ratings_smoothed = torch.tensor(y_smoothed[prev_ids_np]) 
                # real_ratings_smoothed = torch.tensor(y_smoothed[real_ids_np]) 
                # targets = torch.le(prev_ratings_smoothed, real_ratings_smoothed).float()
                # targets = targets.to(self.opt.device)
                
                loss = bce_loss_fun(real_outputs - prev_outputs, targets)
                
                """
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                """
                accumulation_steps = 8
                loss = loss/accumulation_steps

                loss.backward()
                # gradient accumulation
                # https://www.zhihu.com/question/303070254
                if((global_step+1)%accumulation_steps)==0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()   # reset gradient
                    
                

                comp_results = torch.le(prev_outputs, real_outputs, out=None) 
                t_outputs = (comp_results).long()
                n_correct += (t_outputs).sum().item()
                                
                n_total += len(t_outputs)
                loss_total += loss.item() * len(t_outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    # pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)
                    # self.model.train()
                prev_sample_batched = sample_batched

            if global_step % self.opt.log_step != 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                # pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)
                # self.model.train()

            if((global_step+1)%accumulation_steps)!=0:
                # some gradients are not updated yet! 
                optimizer.step()
                optimizer.zero_grad()   # reset gradient

            pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)               
            val_acc, val_f1 = pair_val_acc, pair_val_f1

            # Test accuracy 
            pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   
            # logger.info('> pair_acc: {:.4f}, pair_f1: {:.4f}'.format(pair_acc, pair_f1))
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path


    def _train_stage_2(self, criterion, optimizer, train_data_loader):
        #######################
        # Stage 2: Fine-tuning with smoothed self-reported user ratings
        #######################
        max_val_acc = 0 
        max_val_f1  = 0 
        global_step = 0 
        path = None 

        # cmp_merge_sigmoid = nn.Sigmoid()
        bce_loss_fun = nn.BCEWithLogitsLoss().cuda() # torch.nn.BCELoss().cuda()

        pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   

        for epoch in range(5): #range(self.opt.num_epoch):
            
            cur_trainset_pool = self.trainset
            train_data_loader = DataLoader(dataset=cur_trainset_pool, batch_size=self.opt.batch_size, shuffle=False, drop_last=False)
            self.extract_features(train_data_loader, "epoch" + str(epoch))
            y_smoothed = knn_smooth_scores()

            # # reset the whole model 
            # self._reset_params()
            # self.bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)

            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            optimizer.zero_grad()
            prev_sample_batched = None
            for sample_batched in train_data_loader:
                if prev_sample_batched is None:
                    prev_sample_batched = sample_batched
                    continue
                
                global_step += 1
                # clear gradient accumulators


                # inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                inputs = [[x.to(self.opt.device) for x in sample_batched[col]]  for col in self.opt.inputs_cols]
                
                # generate a fake batch here 
                real_outputs, _ = self.model(inputs)
                real_outputs = real_outputs.squeeze(1)
                

                
                prev_inputs = [[x.to(self.opt.device) for x in prev_sample_batched[col]]  for col in self.opt.inputs_cols]
                prev_outputs, _ = self.model(prev_inputs)
                prev_outputs = prev_outputs.squeeze(1)
                
                
                # prev_ratings = prev_sample_batched['rating_num']
                # real_ratings = sample_batched['rating_num']
                # targets = torch.le(prev_ratings, real_ratings).float()
                # targets = targets.to(self.opt.device)

                prev_ids_np = prev_sample_batched['data_id'].numpy()
                real_ids_np = sample_batched['data_id'].numpy()
                prev_ratings_smoothed = torch.tensor(y_smoothed[prev_ids_np]) 
                real_ratings_smoothed = torch.tensor(y_smoothed[real_ids_np]) 
                targets = torch.le(prev_ratings_smoothed, real_ratings_smoothed).float()
                targets = targets.to(self.opt.device)
                
                loss = bce_loss_fun(real_outputs - prev_outputs, targets)
                
                """
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                """
                accumulation_steps = 8
                loss = loss/accumulation_steps

                loss.backward()
                # gradient accumulation
                # https://www.zhihu.com/question/303070254
                if((global_step+1)%accumulation_steps)==0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()   # reset gradient
                    
                

                comp_results = torch.le(prev_outputs, real_outputs, out=None) 
                t_outputs = (comp_results).long()
                n_correct += (t_outputs).sum().item()
                                
                n_total += len(t_outputs)
                loss_total += loss.item() * len(t_outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    # pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)
                    # self.model.train()
                prev_sample_batched = sample_batched

            if global_step % self.opt.log_step != 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                # pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)
                # self.model.train()

            if((global_step+1)%accumulation_steps)!=0:
                # some gradients are not updated yet! 
                optimizer.step()
                optimizer.zero_grad()   # reset gradient

            pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)               
            val_acc, val_f1 = pair_val_acc, pair_val_f1

            # Test accuracy 
            pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   
            # logger.info('> pair_acc: {:.4f}, pair_f1: {:.4f}'.format(pair_acc, pair_f1))
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path


    def run_stage_1(self):
        #######################
        # Stage 1: Learning Representation viaself-supervised dialog anomaly detection
        #######################
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        logger.info('self.trainset: {}'.format(len(self.trainset)))
        best_model_path = self._train_stage_1(criterion, optimizer, train_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        print(best_model_path)
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=False)
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)
        return best_model_path


    def _train_stage_1(self, criterion, optimizer, train_data_loader):
        #######################
        # Stage 1: Learning Representation viaself-supervised dialog anomaly detection
        #######################
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None

        # cmp_merge_sigmoid = nn.Sigmoid()
        bce_loss_fun = nn.BCEWithLogitsLoss().cuda() # torch.nn.BCELoss().cuda()

        pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   


        for epoch in range(1): #range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_usr_correct, n_sys_correct, n_total, loss_total = 0, 0, 0, 0
            # switch model to training mode
            self.model.train()
            optimizer.zero_grad()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                

                # inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                inputs = [[x.to(self.opt.device) for x in sample_batched[col]]  for col in self.opt.inputs_cols]
                
                # generate a fake batch here 
                real_outputs, _ = self.model(inputs)
                # real_outputs = real_outputs[:,1]
                real_outputs = real_outputs.squeeze(1)
                # real_outputs = 4 * F.sigmoid(real_outputs)
                

                
                fake_usr_batched, fake_sys_batched = self.fake_a_batch(sample_batched, self.trainset)
                # fake_batched = fake_usr_batched
                # fake_batched = fake_sys_batched
                # del fake_usr_batched, fake_sys_batched 

                fake_inputs = [[x.to(self.opt.device) for x in fake_usr_batched[col]]  for col in self.opt.inputs_cols]
                fake_outputs, _ = self.model(fake_inputs)
                # fake_outputs = fake_outputs[:,1]
                fake_usr_outputs = fake_outputs.squeeze(1)
                # fake_outputs = 4 * F.sigmoid(fake_outputs)

                fake_inputs = [[x.to(self.opt.device) for x in fake_sys_batched[col]]  for col in self.opt.inputs_cols]
                fake_outputs, _ = self.model(fake_inputs)
                # fake_outputs = fake_outputs[:,1]
                fake_sys_outputs = fake_outputs.squeeze(1)
                # fake_outputs = 4 * F.sigmoid(fake_outputs)

                # unsupervised setting. targets = sample_batched['polarity'].to(self.opt.device)
                targets = torch.ones_like(sample_batched['polarity'],dtype=torch.float32)
                targets = targets.to(self.opt.device)

                # outputs = cmp_merge_sigmoid(real_outputs - fake_outputs)
                # loss = bce_loss_fun(outputs, targets)
                loss = bce_loss_fun(real_outputs - fake_usr_outputs, targets) + bce_loss_fun(real_outputs - fake_sys_outputs, targets)
                # loss = bce_loss_fun(real_outputs - fake_sys_outputs, targets)

                # loss = criterion(outputs, targets)
                """
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                """
                accumulation_steps = 8
                loss = loss/accumulation_steps

                loss.backward()
                # gradient accumulation
                # https://www.zhihu.com/question/303070254
                if((global_step+1)%accumulation_steps)==0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()   # reset gradient
                    
                

                comp_results = torch.le(fake_usr_outputs, real_outputs, out=None) 
                t_outputs = (comp_results).long()
                n_usr_correct += (t_outputs).sum().item()
                
                comp_results = torch.le(fake_sys_outputs, real_outputs, out=None) 
                t_outputs = (comp_results).long()
                n_sys_correct += (t_outputs).sum().item()
                
                n_total += len(t_outputs)
                loss_total += loss.item() * len(t_outputs)
                if global_step % self.opt.log_step == 0:
                    train_usr_acc = n_usr_correct / n_total
                    train_sys_acc = n_sys_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, usr acc: {:.4f}, sys acc: {:.4f}'.format(train_loss, train_usr_acc, train_sys_acc))
                    # pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)
                    # self.model.train()

            if global_step % self.opt.log_step != 0:
                train_usr_acc = n_usr_correct / n_total
                train_sys_acc = n_sys_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, usr acc: {:.4f}, sys acc: {:.4f}'.format(train_loss, train_usr_acc, train_sys_acc))

            if((global_step+1)%accumulation_steps)!=0:
                # some gradients are not updated yet! 
                optimizer.step()
                optimizer.zero_grad()   # reset gradient

            pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)               
            val_acc, val_f1 = pair_val_acc, pair_val_f1

            # Test accuracy 
            pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   
            # logger.info('> pair_acc: {:.4f}, pair_f1: {:.4f}'.format(pair_acc, pair_f1))
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _extract_pair_annotated_for_shapley_dev(self, isExtractTest=True):
        
        # dataset = self.annotaed_pair_test_dataset # + self.annotaed_pair_val_dataset
        if isExtractTest is True:
            dataset = self.annotaed_pair_test_dataset 
            pkl_path = "./extract/bert_pair_alexa10.pkl"
        else:
            dataset = self.annotaed_pair_val_dataset 
            pkl_path = "./extract/bert_pair_alexa10_dev.pkl"

        pairs_total_num = len(dataset)
        print("_extract_pair_annotated_for_shapley_dev: pairs_total_num", pairs_total_num)
        data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=False)
        

        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                dial1_t_sample_batched = t_sample_batched['dial1']
                dial1_t_inputs = [ [x.to(self.opt.device) for x in dial1_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                _, dial1_t_features = self.model(dial1_t_inputs)
                

                dial2_t_sample_batched = t_sample_batched['dial2']
                dial2_t_inputs = [ [x.to(self.opt.device) for x in dial2_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                _, dial2_t_features = self.model(dial2_t_inputs)

                t_targets = (t_sample_batched['compare_res']-1).to(self.opt.device)
                # print("t_targets", t_targets.shape, t_targets, t_targets.dtype)

                # t_ids = t_sample_batched["data_id"]
                
                
                if t_targets_all is None:
                    t_targets_all = t_targets
                    # t_outputs_all = t_outputs
                    # t_ids_all = t_ids
                    dial1_t_features_all = dial1_t_features
                    dial2_t_features_all = dial2_t_features
                    
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    # t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    # t_ids_all = torch.cat((t_ids_all, t_ids), dim=0)
                    dial1_t_features_all = torch.cat((dial1_t_features_all, dial1_t_features), dim=0)
                    dial2_t_features_all = torch.cat((dial2_t_features_all, dial2_t_features), dim=0)
        
                    t_targets_all_cpu = t_targets_all.cpu().numpy()
            
            dial1_t_features_all = dial1_t_features_all.cpu().numpy()
            dial2_t_features_all = dial2_t_features_all.cpu().numpy()
            # t_ids_all_cpu = t_ids_all.cpu().numpy()
            t_targets_all_cpu = t_targets_all.cpu().numpy()
            
            this_result_dict = {
                # "t_inputs_all_cpu": t_inputs_all, 
                # "t_ids_all_cpu": t_ids_all_cpu,
                "t_targets_all_cpu": t_targets_all_cpu, 
                # "t_outputs_all_cpu": t_outputs_all_cpu,
                "dial1_t_features_all": dial1_t_features_all,
                "dial2_t_features_all": dial2_t_features_all,
            }
            
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(this_result_dict, pkl_file)


        return


    def _pair_tie_auc_evaluate(self):

        # first collect tie outputs
        dataset = self.annotaed_pair_tie_dataset
        data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=False)
        dial12_diff_t_outputs_all = None
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                dial1_t_sample_batched = t_sample_batched['dial1']
                dial1_t_inputs = [ [x.to(self.opt.device) for x in dial1_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial1_t_outputs, _ = self.model(dial1_t_inputs)
                dial1_t_outputs = dial1_t_outputs.squeeze(1)
                dial2_t_sample_batched = t_sample_batched['dial2']
                dial2_t_inputs = [ [x.to(self.opt.device) for x in dial2_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial2_t_outputs, _ = self.model(dial2_t_inputs)
                dial2_t_outputs = dial2_t_outputs.squeeze(1)
                comp_results = torch.le(dial1_t_outputs, dial2_t_outputs, out=None) 
                if dial12_diff_t_outputs_all is None:
                    dial12_diff_t_outputs_all = torch.abs(dial2_t_outputs - dial1_t_outputs)
                else:
                    dial12_diff_t_outputs_all  = torch.cat((dial12_diff_t_outputs_all, torch.abs(dial2_t_outputs - dial1_t_outputs)), dim=0)    
        tie_unnormed_diff_score_pred = dial12_diff_t_outputs_all.cpu().numpy() 

        # then collect non-tie pairs
        dataset = self.annotaed_pair_test_dataset + self.annotaed_pair_val_dataset
        data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=False)
        dial12_diff_t_outputs_all = None
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                dial1_t_sample_batched = t_sample_batched['dial1']
                dial1_t_inputs = [ [x.to(self.opt.device) for x in dial1_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial1_t_outputs, _ = self.model(dial1_t_inputs)
                dial1_t_outputs = dial1_t_outputs.squeeze(1)
                dial2_t_sample_batched = t_sample_batched['dial2']
                dial2_t_inputs = [ [x.to(self.opt.device) for x in dial2_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial2_t_outputs, _ = self.model(dial2_t_inputs)
                dial2_t_outputs = dial2_t_outputs.squeeze(1)
                comp_results = torch.le(dial1_t_outputs, dial2_t_outputs, out=None) 
                if dial12_diff_t_outputs_all is None:
                    dial12_diff_t_outputs_all = torch.abs(dial2_t_outputs - dial1_t_outputs)
                else:
                    dial12_diff_t_outputs_all  = torch.cat((dial12_diff_t_outputs_all, torch.abs(dial2_t_outputs - dial1_t_outputs)), dim=0)    
        nontie_unnormed_diff_score_pred = dial12_diff_t_outputs_all.cpu().numpy() 

        tie_nontie_ground_truth = np.concatenate(
            (np.zeros_like(tie_unnormed_diff_score_pred), np.ones_like(nontie_unnormed_diff_score_pred)), 
            axis=0)
        tie_nontie_pred = np.concatenate((tie_unnormed_diff_score_pred, nontie_unnormed_diff_score_pred), axis=0)
        tie_auc_score = roc_auc_score(tie_nontie_ground_truth, tie_nontie_pred)
        fpr, tpr, thresholds = roc_curve(tie_nontie_ground_truth, tie_nontie_pred)
        # print("tie_nontie_ground_truth, tie_nontie_pred", tie_nontie_ground_truth, tie_nontie_pred)
        logger.info("auc score on tie/ non-tie: {}".format(tie_auc_score))
        # logger.info("fpr, tpr, thresholds: {},{}, {}".format(fpr, tpr, thresholds))
        return 



    def _pair_annotated_evaluate(self, isTestFlag):
        # val or test? 
        if isTestFlag:
            dataset = self.annotaed_pair_test_dataset
        else:
            # is val
            dataset = self.annotaed_pair_val_dataset
        data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        dial12_t_outputs_all, dial12selfratings_all = None, None
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                dial1_t_sample_batched = t_sample_batched['dial1']
                dial1_t_inputs = [ [x.to(self.opt.device) for x in dial1_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial1_t_outputs, _ = self.model(dial1_t_inputs)
                dial1_t_outputs = dial1_t_outputs.squeeze(1)                
                dial2_t_sample_batched = t_sample_batched['dial2']
                dial2_t_inputs = [ [x.to(self.opt.device) for x in dial2_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial2_t_outputs, _ = self.model(dial2_t_inputs)
                dial2_t_outputs = dial2_t_outputs.squeeze(1)
                comp_results = torch.le(dial1_t_outputs, dial2_t_outputs, out=None) 
                t_outputs = (comp_results).long()
                t_targets = (t_sample_batched['compare_res']-1).to(self.opt.device)
                n_correct += (t_outputs == t_targets).sum().item()
                n_total += len(t_targets)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                    dial12_t_outputs_all  = torch.cat((dial1_t_outputs, dial2_t_outputs), dim=0)
                    dial12selfratings_all = torch.cat((dial1_t_sample_batched['rating_num'], dial2_t_sample_batched['rating_num']), dim=0)
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                    dial12_t_outputs_all  = torch.cat((dial12_t_outputs_all, dial1_t_outputs, dial2_t_outputs), dim=0)
                    dial12selfratings_all = torch.cat((dial12selfratings_all, dial1_t_sample_batched['rating_num'], 
                    dial2_t_sample_batched['rating_num']), dim=0) # should alwys on the cpu

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2, 3, 4, 5], average='macro')
        
        unnormed_score_pred = dial12_t_outputs_all.cpu().numpy() 
        noisy_usr_rating = dial12selfratings_all.numpy()  

        rho, pval = spearmanr(unnormed_score_pred, noisy_usr_rating)
        logger.info('>> spearmanr rho, pval:{}, {}'.format(rho, pval))
        
        rho, pval = pearsonr(unnormed_score_pred, noisy_usr_rating)
        logger.info('>> pearsonr rho, pval:{}, {}'.format(rho, pval))
        

        pairs_total_num = len(dataset) 
        if isTestFlag:
            logger.info('>> [Test] pairs_total_num:{}, pair_acc: {:.4f}, pair_f1: {:.4f}'.format(pairs_total_num, acc, f1))
        else:
            logger.info('>> [Val] pairs_total_num:{}, pair_acc: {:.4f}, pair_f1: {:.4f}'.format(pairs_total_num, acc, f1))    
        self._pair_tie_auc_evaluate()
        return acc, f1



    def _pair_annotated_train_on_dev(self, optimizer, isTestFlag=False):
        
        # val or test? 
        if isTestFlag:
            dataset = self.annotaed_pair_test_dataset
            assert 0
        else:
            # is val
            dataset = self.annotaed_pair_val_dataset

        data_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, shuffle=True)
        
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   

        bce_loss_fun = nn.BCEWithLogitsLoss().cuda() # torch.nn.BCELoss().cuda()


        # switch model to evaluation mode
        self.model.train()
        # with torch.no_grad():
        for epoch in range(5): 
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            optimizer.zero_grad()
            for t_batch, t_sample_batched in enumerate(data_loader):
                dial1_t_sample_batched = t_sample_batched['dial1']
                dial1_t_inputs = [ [x.to(self.opt.device) for x in dial1_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial1_t_outputs, _ = self.model(dial1_t_inputs)
                # dial1_t_outputs = F.softmax(dial1_t_outputs, dim=1)
                # dial1_t_outputs = dial1_t_outputs[:,1]
                dial1_t_outputs = dial1_t_outputs.squeeze(1)
                
                dial2_t_sample_batched = t_sample_batched['dial2']
                dial2_t_inputs = [ [x.to(self.opt.device) for x in dial2_t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                dial2_t_outputs, _ = self.model(dial2_t_inputs)
                # dial2_t_outputs = F.softmax(dial2_t_outputs, dim=1)
                # dial2_t_outputs = dial2_t_outputs[:,1]
                dial2_t_outputs = dial2_t_outputs.squeeze(1)
                
                
                t_targets = (t_sample_batched['compare_res']-1).to(self.opt.device)
                loss = bce_loss_fun(dial2_t_outputs - dial1_t_outputs, t_targets.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()   # reset gradient
                    
                

                comp_results = torch.le(dial1_t_outputs, dial2_t_outputs, out=None) 
                t_outputs = (comp_results).long()
                n_correct += (t_outputs == t_targets).sum().item()
                n_total += len(t_targets)
                loss_total += loss.item() * len(t_outputs)
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            logger.info('>> [overfit] on pairs eval, ONLY AS TRAIN ACC.')
            pair_val_acc, pair_val_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
            val_acc, val_f1 = pair_val_acc, pair_val_f1
            
            pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   
            
            if not os.path.exists('state_dict'):
                os.mkdir('state_dict')
            path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(pair_acc, 4))
            torch.save(self.model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
            
            # break

        return path


    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                # t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # print("t_sample_batched", t_sample_batched)
                t_inputs = [ [x.to(self.opt.device) for x in t_sample_batched[col]  ] for col in self.opt.inputs_cols]
                # print("t_inputs", t_inputs)
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        # f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2, 3, 4, 5], average='macro')
        
        y_test = t_targets_all.cpu()
        pre_test = torch.argmax(t_outputs_all, -1).cpu()
        print("confusion_matrix\n", confusion_matrix(y_test,pre_test))
        print("classification_report\n", classification_report(y_test,pre_test))
        print("accuracy_score\n", accuracy_score(y_test, pre_test))

        return acc, f1


    def extract_features(self, train_data_loader, iter_step, train_dataset=None): 
        
        if train_dataset is None:
            train_dataset = self.trainset
        self._pair_annotated_evaluate(isTestFlag=True)
        self._extract_pair_annotated_for_shapley_dev(isExtractTest=True)
        self._extract_pair_annotated_for_shapley_dev(isExtractTest=False) # Extract DEV set
        this_dir = './extract'
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)

        extractor_tasks = [
            # [self.valset, val_data_loader, "./extract/bert_alexa10_val.pkl"],
            [train_dataset, train_data_loader, "./extract/bert_alexa10_train.pkl"],
            # [self.testset, test_data_loader, "./extract/bert_alexa10_test.pkl"],
        ]

        for dataset, data_loader, pkl_path in extractor_tasks:
            n_correct, n_total = 0, 0
            t_targets_all, t_outputs_all = None, None

            t_inputs_all, t_ids_all, t_features_all = None, None, None
            # switch model to evaluation mode
            list_text_left_list = []
            list_text_right_list = []
            rating_list = []
            conversation_id_list = []
            self.model.eval()
            with torch.no_grad():
                for t_batch, t_sample_batched in enumerate(data_loader):

                    data_ids_list = t_sample_batched['data_id'].numpy().tolist()
                    for i, data_id in enumerate(data_ids_list):
                        datapoint = dataset[data_id]
                        text_left_list = datapoint['text_left_list']
                        text_right_list = datapoint['text_right_list']
                        list_text_left_list.append(text_left_list) # in batch
                        list_text_right_list.append(text_right_list)
                        rating_list.append(datapoint['rating'])
                        conversation_id_list.append(datapoint['conversation_id'])

                    t_inputs = [[x.to(self.opt.device) for x in t_sample_batched[col]]  for col in self.opt.inputs_cols]
                    t_targets = t_sample_batched['polarity'].to(self.opt.device)
                    t_outputs, t_features = self.model(t_inputs)
                    t_ids = t_sample_batched["data_id"]
                    n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_total += len(t_outputs)

                    if t_targets_all is None:
                        t_targets_all = t_targets
                        t_outputs_all = t_outputs
                        t_ids_all = t_ids
                        t_features_all = t_features
                    else:
                        t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                        t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                        t_ids_all = torch.cat((t_ids_all, t_ids), dim=0)
                        t_features_all = torch.cat((t_features_all, t_features), dim=0)
                    

            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
            logger.info('> {} acc: {:.4f}, f1: {:.4f}'.format(pkl_path, acc, f1))

            t_targets_all_cpu = t_targets_all.cpu().numpy()
            t_outputs_all_cpu = t_outputs_all.cpu().numpy()
            t_ids_all_cpu = t_ids_all.cpu().numpy()
            t_features_all_cpu = t_features_all.cpu().numpy() 

            this_result_dict = {
                "t_ids_all_cpu": t_ids_all_cpu,
                "t_targets_all_cpu": t_targets_all_cpu, 
                "t_outputs_all_cpu": t_outputs_all_cpu,
                "t_features_all_cpu": t_features_all_cpu,

                "list_text_left_list": list_text_left_list,  
                "list_text_right_list": list_text_right_list,  
                'rating': rating_list, 
                'conversation_id': conversation_id_list, 
            }

            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(this_result_dict, pkl_file)

        return 



    def pipeline_3_stages(self):
        
        #######################
        # Stage 1: Learning Representation viaself-supervised dialog anomaly detection
        #######################
        checkpoint_path = self.run_stage_1() 
        logger.info('stage_1 checkpoint_path: {}'.format(checkpoint_path))        
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        #######################
        # Stage 2: Fine-tuning with smoothed self-reported user ratings
        #######################
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True)
        logger.info('self.trainset: {}'.format(len(self.trainset)))
        checkpoint_path = self._train_stage_2(criterion, optimizer, train_data_loader)
        logger.info('_train_stage_2 checkpoint_path: {}'.format(checkpoint_path))
        self.model.load_state_dict(torch.load(checkpoint_path), strict=True) 
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   
        
        #######################
        # Stage 3: Denoising with data Shapley &further fine-tuning 
        #######################
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        self.extract_kmeans_newset(train_data_loader, "shapley")
        global_shapley_arr = do_knn_shapley() # always do it in a global manner

        next_trainset_pool = ABSADataset()
        for i in range(global_shapley_arr.shape[0]):
            if global_shapley_arr[i] > 0.:
                next_trainset_pool.data.append(self.trainset.data[i])                

        logger.info('After shapley: len(next_trainset_pool.data) {}, global_shapley_arr {}'.format( len(next_trainset_pool.data), global_shapley_arr ))
        train_data_loader = DataLoader(dataset=next_trainset_pool, batch_size=self.opt.batch_size, shuffle=True, drop_last=True)
        best_model_path = self._train_stage_3(criterion, optimizer, train_data_loader)
        logger.info('After shapley best_model_path: {}'.format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=False)   
        pair_acc, pair_f1 = self._pair_annotated_evaluate(isTestFlag=True)   



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=4, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--polarities_dim', default=1, type=int) 
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_spc': BERT_SPC,
    }
    
    input_colses = {
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]        
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = './log/{}-{}.log'.format(opt.model_name,strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.pipeline_3_stages() # main algorithm 

from torch.autograd import Variable


class BERT_SPC(nn.Module): # save 
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.opt_hidded_dim = opt.hidden_dim


    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0][0], inputs[1][0]
        _, pooled_output_origin = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output_origin)
        logits = self.dense(pooled_output)
        return logits, pooled_output_origin


if __name__ == "__main__":
    main()

