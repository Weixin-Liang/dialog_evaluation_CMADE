"""
To run the training pipeline of the bert sentence classifier, please run: 
    CUDA_VISIBLE_DEVICES=0 python bertspc.py
"""

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy  as np
from transformers import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data_utils import Tokenizer4Bert, ABSADataset_window
import pickle
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve, classification_report, confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        self.tokenizer = tokenizer

        pkl_path = "./datacache_bertspc.pkl"
        regenerate_data_flag = True  
        if not regenerate_data_flag and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as pkl_file:
                data_dict = pickle.load(pkl_file)
                self.train_all = data_dict['self.train_all']
                self.trainset = data_dict['self.trainset']
                self.eval_all = data_dict['self.eval_all']
                self.evalset = data_dict['self.evalset']
                self.test_all = data_dict['self.test_all']
                self.testset = data_dict['self.testset']
                self.testdevset = data_dict['self.testdevset']
        else:
            # self.load_gunrock_dataset()
            self.load_convai_dataset()

            with open(pkl_path, 'wb') as pkl_file:
                data_dict = {
                    "self.train_all": self.train_all,
                    "self.trainset": self.trainset,
                    "self.eval_all": self.eval_all,
                    "self.evalset": self.evalset,
                    "self.test_all": self.test_all,
                    "self.testset": self.testset,
                    "self.testdevset": self.testdevset,
                }
                pickle.dump(data_dict, pkl_file)


        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt)
        self.model = self.model.to(opt.device)


        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()


    def load_convai_dataset(self):

        from convai_data.convai_dataloader import load_extract_full_dataset, load_annotated_eval_dataset
        self.test_all = load_annotated_eval_dataset()
        annotated_non_low_windows, annotated_low_engagement_windows = self.test_all

        print(
            "ConvAI len(annotated_low_engagement_windows)",
            len(annotated_low_engagement_windows), 
            )

        random.seed(42)
        random.shuffle(annotated_non_low_windows)
        random.shuffle(annotated_low_engagement_windows)
        print("ConvAI len(annotated_non_low_windows)", len(annotated_non_low_windows) )

        self.testset = ABSADataset_window(
            (
                annotated_non_low_windows[1::2],
                annotated_low_engagement_windows[1::2],
            ),
            self.tokenizer,
            balanced_upsampling_flag=True, 
            )

        self.testdevset = ABSADataset_window(
            (
                annotated_non_low_windows[0::2],
                annotated_low_engagement_windows[0::2],
            ),
            self.tokenizer,
            balanced_upsampling_flag=True 
            )
        auto_labeled_non_low_windows, auto_labeled_low_engagement_window = load_extract_full_dataset()

        random.seed(42)
        random.shuffle(auto_labeled_non_low_windows)
        random.shuffle(auto_labeled_low_engagement_window)

        print("ConvAI auto_labeled_non_low_windows", len(auto_labeled_non_low_windows),
        "ConvAI auto_labeled_low_engagement_window", len(auto_labeled_low_engagement_window))
        self.train_all = auto_labeled_non_low_windows[0::10], auto_labeled_low_engagement_window[0::10]

        self.trainset = ABSADataset_window(self.train_all, self.tokenizer)
        self.trainset.data.extend(self.testdevset.data)
        self.eval_all = auto_labeled_non_low_windows[-500:], auto_labeled_low_engagement_window[-500:]
        self.evalset = ABSADataset_window(self.eval_all, self.tokenizer)
        return


    def load_gunrock_dataset(self):
        from gunrock_data.dataio import load_extract_full_dataset
        from gunrock_data.old_dataio import load_annotated_eval_dataset


        self.test_all = load_annotated_eval_dataset()
        annotated_high_engagement_windows, annotated_mid_engagement_windows, annotated_low_engagement_windows = self.test_all
        print("annotated_high_engagement_windows", len(annotated_high_engagement_windows), "annotated_mid_engagement_windows", len(annotated_mid_engagement_windows), "annotated_low_engagement_windows", len(annotated_low_engagement_windows) )

        self.testset = ABSADataset_window(
            (
                annotated_high_engagement_windows[0::2] + annotated_mid_engagement_windows[0::2],
                annotated_low_engagement_windows[0::2],
            ),
            self.tokenizer,
            balanced_upsampling_flag=True
            )

        self.testdevset = ABSADataset_window(
            (
                annotated_high_engagement_windows[1::4] + annotated_mid_engagement_windows[1::4],
                annotated_low_engagement_windows[1::4],
            ),
            self.tokenizer,
            balanced_upsampling_flag=True
            )


        auto_labeled_non_low_windows, auto_labeled_low_engagement_window = load_extract_full_dataset()
        self.train_all = auto_labeled_non_low_windows[:1000], auto_labeled_low_engagement_window[:1000]
        self.trainset = ABSADataset_window(self.train_all, self.tokenizer)
        self.eval_all = auto_labeled_non_low_windows[-500:], auto_labeled_low_engagement_window[-500:]
        self.evalset = ABSADataset_window(self.eval_all, self.tokenizer)
        return


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
            if type(child) != BertModel:  
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)


    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_acc = 0
        max_val_f1 = 0

        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            global_step = 0
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, _ = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 1:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if True: 
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

            test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
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
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')


        y_test = t_targets_all.cpu()
        pre_test = torch.argmax(t_outputs_all, -1).cpu()
        print("confusion_matrix\n", confusion_matrix(y_test,pre_test))
        print("classification_report\n", classification_report(y_test,pre_test))
        print("accuracy_score\n", accuracy_score(y_test, pre_test))


        return acc, f1


    def fine_tune_on_denoised(self):
        self.model.load_state_dict(torch.load("gunrock_state_dict/autolabel2.cpt")) 
        shapley_pkl_path = './shapleyOut/shapley_value.pkl'
        with open(shapley_pkl_path, 'rb') as pkl_file:
            data_dict = pickle.load(pkl_file)
            global_shapley_arr = data_dict['global_shapley_arr']
            y_loaded = data_dict['y']

        import copy
        tmp_trainset = copy.deepcopy(self.trainset)
        tmp_trainset_reverse = copy.deepcopy(self.trainset)
        for datum_to_reverse in tmp_trainset_reverse.data:
            datum_to_reverse['polarity'] = 1 - datum_to_reverse['polarity']
        origin_train_len = len(self.trainset)

        concat_data_list = tmp_trainset.data + tmp_trainset_reverse.data
        new_data_list = []
        for i in range(2 * origin_train_len):
            if global_shapley_arr[i] > 0.:
                datum = concat_data_list[i]
                new_data_list.append(datum)

        self.trainset.data = new_data_list

        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.evalset, batch_size=self.opt.batch_size * 4, shuffle=False) 
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size * 4, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        return


    def run(self):

        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.evalset, batch_size=self.opt.batch_size * 4, shuffle=False) 
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size * 4, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int) 
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
        'adadelta': torch.optim.Adadelta,  
        'adagrad': torch.optim.Adagrad,  
        'adam': torch.optim.Adam,  
        'adamax': torch.optim.Adamax,  
        'asgd': torch.optim.ASGD, 
        'rmsprop': torch.optim.RMSprop, 
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
    ins.run() 
    # ins.fine_tune_on_denoised()



from torch.autograd import Variable


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits, pooled_output

if __name__ == "__main__":
    main()