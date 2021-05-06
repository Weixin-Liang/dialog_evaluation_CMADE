import csv
import pandas as pd

def load_extract_full_dataset():
    # load auto-labled training set
    input_file = 'convai_data/training_set/group_0.csv'
    return feature_regenerator_for_conv_ai_data(input_file)

def load_annotated_eval_dataset():
    # load human annotated set
    # input_file = 'convai_data/ConvAI-annotation.csv'
    input_file = 'convai_data/ConvAI2_ReAnnotated.csv'
    return feature_regenerator_for_conv_ai_data(input_file)


def feature_regenerator_for_conv_ai_data(input_file):
    data = load_data(input_file)

    annotated_low_engagement_windows = []
    non_low_windows = []

    dialog_count = 0

    last_bot_response = "EMPTY"
    for row in data:
        if row["turn_id"] == '0': # new conversation
            last_bot_response = "EMPTY"
            dialog_count += 1
            continue # skip the first turn

        if row["sender_class"] == "Bot":
            last_bot_response = row["text"]
        if row["sender_class"] == "Human":
            # print("sender_class is human! ")

            #######################
            # Construct windows
            #######################
            window = {
                'last_response': [  last_bot_response  ],
                'text': [ row["text"] ],
                # does not really apply here
                'conversation_id': str(row['participant1_id']) + str(row['participant2_id']),
                'turn_j': int(row['turn_id']),
            }
            assert "engagement (human_annotated)" in row or "engagement (auto-labeled)" in row
            if "engagement (human_annotated)" in row:
                engagement_col_name = "engagement (human_annotated)"
            else:
                engagement_col_name = "engagement (auto-labeled)"

            if row[engagement_col_name] == '0':
                non_low_windows.append(window)

            elif row[engagement_col_name] == '-1':
                annotated_low_engagement_windows.append(window)

    print("len(annotated_low_engagement_windows)", len(annotated_low_engagement_windows))
    print("len(non_low_windows)", len(non_low_windows) )
    print("dialog_count", dialog_count)
    return non_low_windows, annotated_low_engagement_windows



def load_data(file):
    with open(file) as input_file:
        a = [{k: v for k, v in row.items()} for row in csv.DictReader(input_file, skipinitialspace=True)]
    return a

if __name__ == '__main__':
    # input_file = 'ConvAI2_ReAnnotated.csv'
    input_file = 'training_set/group_0.csv'
    feature_regenerator_for_conv_ai_data(input_file)
