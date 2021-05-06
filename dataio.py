import pandas as pd
import os
from collections import Counter 
import re
import json 
import math
import matplotlib.pyplot as plt 
import time
import csv



def load_annotated_tie_pairs(meta_annotation_csv_path):
    ret = []
    used_dialog_ids = set()
    with open(meta_annotation_csv_path, "r", encoding='utf-8') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(csv_file):
            if i == 0:
                continue
            new_row = dict()
            new_row['annotation_pair_id'] = row[0]
            new_row['dial1_id'] = row[1]
            new_row['dial2_id'] = row[2]
            ret.append(new_row)    
            used_dialog_ids.add(new_row['dial1_id'])
            used_dialog_ids.add(new_row['dial2_id'])
            
    return ret, used_dialog_ids


def load_annotated_pairs(meta_annotation_csv_path="./morepairs218.csv"):
    ret = []
    used_dialog_ids = set()
    with open(meta_annotation_csv_path, "r", encoding='utf-8') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(csv_file):
            if i == 0:
                continue
            new_row = dict()
            new_row['annotation_pair_id'] = row[0]
            new_row['dial1_id'] = row[1]
            new_row['dial2_id'] = row[2]
            new_row['compare_res'] = int(row[3])
            assert(new_row['compare_res'] == 1 or new_row['compare_res'] == 2)
            ret.append(new_row)    
            used_dialog_ids.add(new_row['dial1_id'])
            used_dialog_ids.add(new_row['dial2_id'])
            
    return ret, used_dialog_ids



def load_dataset(data_set_csv_file_name, used_dialog_ids=set(), early_truncate=-1):
    
    dataset = [] 
    total_dial_count = 0
    valid_dial_count = 0

    begin_row_this_dialog = 0
    end_row_this_dialog = 0
    utters_this_dialog = []
    
    def handle_dialog():
        retFlag = True
        each_dialog_data = dict()

        dial_len = end_row_this_dialog - begin_row_this_dialog + 1
        if dial_len == 0:
            return False, None

        if utters_this_dialog[0]['conversation_id'] in used_dialog_ids:
            print("test data found! ignore:", utters_this_dialog[0]['conversation_id'])
            return False, None 


        rating = utters_this_dialog[0]['rating']
        each_dialog_data['rating'] = int(round(rating))
        each_dialog_data['conversation_id'] = utters_this_dialog[0]['conversation_id']
        each_dialog_data['response'] = []
        each_dialog_data['text'] = []

        for sub_cur_utter in utters_this_dialog:
            response = sub_cur_utter['response']
            
            if type(response) is str:
                response = response.encode('ascii', 'ignore').decode('ascii')
                try:
                    response_without_angle_brackets = re.sub(u'<.*?>', '', response) 
                except Exception as e:
                        print(response, utters_this_dialog)
                        print(e)
                        exit(0)
                assert len(response_without_angle_brackets)>0, response_without_angle_brackets
                each_dialog_data['response'].append(response_without_angle_brackets) 
            
            if  type(sub_cur_utter['text']) is str:
                assert len(sub_cur_utter['text'])>0, sub_cur_utter['text']
                text = sub_cur_utter['text'].encode('ascii', 'ignore').decode('ascii')
                each_dialog_data['text'].append(text)


        return retFlag, each_dialog_data

    start_time = time.time()


    filepath = data_set_csv_file_name 
    print("reading data " + filepath)
    df = pd.read_csv(filepath, low_memory=False)

    for j, cur_utter in df.iterrows():
        if early_truncate!=-1 and total_dial_count>early_truncate:
            break
        else:
            pass

        if j == 0:
            prev_id = cur_utter['conversation_id'] 
            prev_module = cur_utter['selected_modules'] 
            prev_response = cur_utter['response'] 

        if j%10000 == 9999:
            print("current progress: ", j, "time elapsed:", time.time()-start_time, "valid_dial_count", valid_dial_count ) 
            
        if prev_id != cur_utter['conversation_id']: 
            end_row_this_dialog = j -1    
            dial_valid_flag, each_dialog_data = handle_dialog()
            if dial_valid_flag is True:
                valid_dial_count += 1
                dataset.append(each_dialog_data)
                
            prev_id = cur_utter['conversation_id']
            prev_module = cur_utter['selected_modules']
            prev_response = cur_utter['response']
            begin_row_this_dialog = j
            total_dial_count += 1
            utters_this_dialog = []
        else:
            pass
        utters_this_dialog.append(cur_utter)

    print("total sentences here: ", j, "total_dial_count: ", total_dial_count, "valid_dial_count", valid_dial_count)
    end_time = time.time()
    print(end_time - start_time)

    return dataset



def balanced_dataset_load(data_set_csv_file_name, used_dialog_ids=set(), early_truncate=-1):
    
    datasets = [ [] for _ in range(6) ] 

    total_dial_count = 0
    valid_dial_count = 0

    begin_row_this_dialog = 0
    end_row_this_dialog = 0
    utters_this_dialog = []
    
    def handle_dialog():
        retFlag = True
        each_dialog_data = dict()

        # dialog len in turns
        dial_len = end_row_this_dialog - begin_row_this_dialog + 1
        if dial_len == 0:
            return False


        if utters_this_dialog[0]['conversation_id'] in used_dialog_ids:
            print("test data found! ignore:", utters_this_dialog[0]['conversation_id'])
            return False, None 

        rating = utters_this_dialog[0]['rating']
        each_dialog_data['rating'] = int(round(rating))
        each_dialog_data['conversation_id'] = utters_this_dialog[0]['conversation_id']
        each_dialog_data['response'] = []
        each_dialog_data['text'] = []

        for sub_cur_utter in utters_this_dialog:
            response = sub_cur_utter['response']
            
            if type(response) is str:
                response = response.encode('ascii', 'ignore').decode('ascii')
                try:
                    response_without_angle_brackets = re.sub(u'<.*?>', '', response) 
                except Exception as e:
                        print(response, utters_this_dialog)
                        print(e)
                        exit(0)
                assert len(response_without_angle_brackets)>0, response_without_angle_brackets
                each_dialog_data['response'].append(response_without_angle_brackets) 
            
            if  type(sub_cur_utter['text']) is str:
                assert len(sub_cur_utter['text'])>0, sub_cur_utter['text']
                text = sub_cur_utter['text'].encode('ascii', 'ignore').decode('ascii')
                each_dialog_data['text'].append(text)


        return retFlag, each_dialog_data

    start_time = time.time()

    filepath = data_set_csv_file_name 
    print("reading data " + filepath)
    df = pd.read_csv(filepath, low_memory=False)

    for j, cur_utter in df.iterrows():
        if early_truncate!=-1 and total_dial_count>early_truncate:
            break

        if j == 0:
            prev_id = cur_utter['conversation_id'] 
            prev_module = cur_utter['selected_modules'] 
            prev_response = cur_utter['response'] 

        if j%10000 == 9999:
            print("current progress: ", j, "time elapsed:", time.time()-start_time, "valid_dial_count", valid_dial_count ) 

        if prev_id != cur_utter['conversation_id']: 
            end_row_this_dialog = j -1    
            dial_valid_flag, each_dialog_data = handle_dialog()
            if dial_valid_flag is True:
                dial_rating = each_dialog_data['rating']
                valid_dial_count += 1
                datasets[dial_rating].append(each_dialog_data)
                
            prev_id = cur_utter['conversation_id']
            prev_module = cur_utter['selected_modules']
            prev_response = cur_utter['response']
            begin_row_this_dialog = j
            total_dial_count += 1
            utters_this_dialog = []
        else:
            pass
        utters_this_dialog.append(cur_utter)

    print("total sentences here: ", j, "total_dial_count: ", total_dial_count, "valid_dial_count", valid_dial_count)
    end_time = time.time()
    print(end_time - start_time)
    print("1,2,3,4,5 len:", end=' ')
    for dataset in datasets:
        print(len(dataset))

    return datasets


def get_all_data(datasets):
    
    ret = []

    for dataset in datasets:
        ret += dataset

    return ret 

def balanced_split_dataset(datasets):
    DEBUG_IO_FLAG = False 
    scores_wanted =  {1,2,3,4,5} 
    train, dev, test = [], [], []

    if DEBUG_IO_FLAG is False:
        for s in scores_wanted:
            test.extend(datasets[s][:125])  
            dev.extend(datasets[s][125:250]) 
            train.extend(datasets[s][250:]) 

    else:
        for s in scores_wanted:
            test.extend(datasets[s][:8])  
            dev.extend(datasets[s][8:16]) 
            train.extend(datasets[s][16:32]) 
    return train, dev, test