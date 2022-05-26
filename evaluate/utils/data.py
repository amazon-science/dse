import os
import csv
import json
import random
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import normalize

from .get_embeddings import calculate_embedding


class SingleSetenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, seq_labels, ner_labels=None):
        self.encodings = encodings
        self.seq_labels = seq_labels
        self.ner_labels = ner_labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['seq_labels'] = torch.tensor(self.seq_labels[idx])
        item['ner_labels'] = torch.tensor(self.ner_labels[idx]) if self.ner_labels else []
        return item

    def __len__(self):
        return len(self.seq_labels)



class SetencePairDataset(torch.utils.data.Dataset):
    def __init__(self, context_encodings, response_encodings, seq_labels=None):
        self.context_encodings = context_encodings
        self.response_encodings = response_encodings
        self.seq_labels = seq_labels

    def __getitem__(self, idx):
        item = {}
        item['context'] = {key: val[idx] for key, val in self.context_encodings.items()}
        item['response'] = {key: val[idx] for key, val in self.response_encodings.items()}
        item['seq_labels'] = torch.tensor(self.seq_labels[idx]) if self.seq_labels else 0
        return item

    def __len__(self):
        return self.context_encodings['input_ids'].shape[0]


'''
Load processed nli data

Input:
    file_path: the path to the folder that contains NLI file. Assume it as a tsv file with three columns: sentence1, sentence2, label

Output:
    texts and labels for train/validation/test sets

'''
def _read_data_nli(file_path):
    if os.path.exists(os.path.join(file_path, "test.txt")):
        test_file_path = file_path
    else:
        test_file_path = "/".join(file_path.rstrip("/").split("/")[:-2])
    
    assert os.path.exists(os.path.join(test_file_path, "test.txt")), "No test file at {} or {}".format(file_path, test_file_path)

    def _load_txt(file_path):
        with open(file_path, "r") as f:
            seq = f.readlines()
            seq = [s.strip("\n").split("\t") for s in seq]
            text = [s[:2] for s in seq]
            seq_labels = [int(s[2]) for s in seq]
        
        return text, seq_labels

    train_text, train_seq_labels = _load_txt(os.path.join(file_path, "train.txt"))
    test_text, test_seq_labels = _load_txt(os.path.join(test_file_path, "test.txt"))
    val_text, val_seq_labels = _load_txt(os.path.join(file_path, "valid.txt"))

    return train_text, test_text, val_text, train_seq_labels, test_seq_labels, val_seq_labels



'''
Pre-process and tokenize nli data

Input:
    BERT_MODEL: model used for training
    file_path: the path to the folder that contains NLI file. Assume it as a tsv file with three columns: sentence1, sentence2, label

Output:
    3 torch.utils.data.Dataset that respectively contains the texts and labels for train/validation/test sets
    number of categories
    weights of each categories (set as the same in current version).

'''
def get_nli_dataset(BERT_MODEL="bert-base-uncased", file_path="data/atis", max_seq_length=100):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    train_text, test_text, val_text, train_seq_labels, test_seq_labels, val_seq_labels = _read_data_nli(file_path)

    unique_seq_labels = set(train_seq_labels).union(test_seq_labels)

    max_len = 0
    for text in train_text + val_text + test_text:
        max_len = max(max_len, len(text[0].split()))
        max_len = max(max_len, len(text[1].split()))
    
    print("Number of training samples: ", len(train_text))
    print("Number of validation samples: ", len(val_text))
    print("Number of test samples: ", len(test_text))
    print("Number of categories: ", len(unique_seq_labels))
    print("Max sequence length: ", max_len)

    train_text_1 = [t[0] for t in train_text] 
    train_text_2 = [t[1] for t in train_text] 
    val_text_1 = [t[0] for t in val_text] 
    val_text_2 = [t[1] for t in val_text] 
    test_text_1 = [t[0] for t in test_text] 
    test_text_2 = [t[1] for t in test_text] 

    train_encodings_1 = tokenizer.batch_encode_plus(train_text_1, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)
    train_encodings_2 = tokenizer.batch_encode_plus(train_text_2, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)
    val_encodings_1 = tokenizer.batch_encode_plus(val_text_1, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)
    val_encodings_2 = tokenizer.batch_encode_plus(val_text_2, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)
    test_encodings_1 = tokenizer.batch_encode_plus(test_text_1, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)
    test_encodings_2 = tokenizer.batch_encode_plus(test_text_2, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length)


    train_dataset = SetencePairDataset(train_encodings_1, train_encodings_2, train_seq_labels)
    val_dataset = SetencePairDataset(val_encodings_1, val_encodings_2, val_seq_labels)
    test_dataset = SetencePairDataset(test_encodings_1, test_encodings_2, test_seq_labels)

    return train_dataset, test_dataset, val_dataset, len(unique_seq_labels), torch.ones([len(unique_seq_labels)])
    




def _read_data_intent_slot(file_path, valid=False, ner=False):
    train_ner, test_ner, val_ner = [[]], [[]], [[]]
    if os.path.exists(os.path.join(file_path, "seq_test.txt")):
        test_file_path = file_path
    else:
        test_file_path = "/".join(file_path.rstrip("/").split("/")[:-2])
    
    assert os.path.exists(os.path.join(test_file_path, "seq_test.txt")), "No test file at {} or {}".format(file_path, test_file_path)

    if ner:
        with open(os.path.join(file_path, "ner_train.txt"), "r") as f:
            train_ner = f.readlines()
            train_ner = [n.strip("\n").split() for n in train_ner]
        with open(os.path.join(test_file_path, "ner_test.txt"), "r") as f:
            test_ner = f.readlines()
            test_ner = [n.strip("\n").split() for n in test_ner]
    with open(os.path.join(file_path, "seq_train.txt"), "r") as f:
        seq_train = f.readlines()
        seq_train = [s.strip("\n").split("\t") for s in seq_train]
        seq_train = [s for s in seq_train if len(s) > 1]
        train_text = [s[0] for s in seq_train]
        train_seq_labels = [int(s[1]) for s in seq_train]
    with open(os.path.join(test_file_path, "seq_test.txt"), "r") as f:
        seq_test = f.readlines()
        seq_test = [s.strip("\n").split("\t") for s in seq_test]
        seq_test = [s for s in seq_test if len(s) > 1]
        test_text = [s[0] for s in seq_test]
        test_seq_labels = [int(s[1]) for s in seq_test]

    if valid:
        if ner:
            with open(os.path.join(file_path, "ner_val.txt"), "r") as f:
                val_ner = f.readlines()
                val_ner = [n.strip("\n").split() for n in val_ner]
        with open(os.path.join(file_path, "seq_val.txt"), "r") as f:
            seq_val = f.readlines()
            seq_val = [s.strip("\n").split("\t") for s in seq_val]
            val_text = [s[0] for s in seq_val]
            val_seq_labels = [int(s[1]) for s in seq_val]

        return train_ner, test_ner, val_ner, train_text, test_text, val_text, train_seq_labels, test_seq_labels, val_seq_labels
    
    else:
        return train_ner, test_ner, train_text, test_text, train_seq_labels, test_seq_labels



def _encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        max_len = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels[:max_len]
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels



def get_intent_slot_dataset(BERT_MODEL="bert-base-uncased", data_path="data/atis", max_seq_length=100, TASK="seq"):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    train_ner, test_ner, val_ner, train_text, test_text, val_text, train_seq_labels, test_seq_labels, val_seq_labels = _read_data_intent_slot(data_path, valid=True, ner=TASK=="ner")
    id2tag = {}

    if TASK=="ner":
        train_ner_tags = set(tag for doc in train_ner for tag in doc)
        val_ner_tags = set(tag for doc in val_ner for tag in doc)
        test_ner_tags = set(tag for doc in test_ner for tag in doc)

        not_seen_slots = []
        for tag in test_ner_tags:
            if tag not in train_ner_tags:
                not_seen_slots.append(tag)
        print("Not seen slots: ", not_seen_slots)

        unique_tags = list(train_ner_tags.union(test_ner_tags).union(val_ner_tags))
        tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
        id2tag = {id: tag for tag, id in tag2id.items()}

        print("Number of unique slots: ", len(id2tag))

    unique_seq_labels = set(train_seq_labels).union(test_seq_labels)

    max_len = 0
    for text in train_text + val_text + test_text:
        max_len = max(max_len, len(text.split()))

    
    print("Number of samples: ", len(train_text))
    print("Number of intents: ", len(unique_seq_labels))
    print("Max sequence length: ", max_len)


    train_encodings = tokenizer(train_text, return_tensors='pt', is_split_into_words=False, return_offsets_mapping=True, padding='longest', truncation=True, max_length=max_seq_length)
    val_encodings = tokenizer(val_text, return_tensors='pt', is_split_into_words=False, return_offsets_mapping=True, padding='longest', truncation=True, max_length=max_seq_length)
    test_encodings = tokenizer(test_text, return_tensors='pt', is_split_into_words=False, return_offsets_mapping=True, padding='longest', truncation=True, max_length=max_seq_length)

    if TASK=="ner":
        train_ner_labels = _encode_tags(train_ner, train_encodings, tag2id)
        val_ner_labels = _encode_tags(val_ner, val_encodings, tag2id)
        test_ner_labels = _encode_tags(test_ner, test_encodings, tag2id)
    else:
        train_ner_labels = None
        val_ner_labels = None
        test_ner_labels = None

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    train_dataset = SingleSetenceDataset(train_encodings, train_seq_labels, train_ner_labels)
    val_dataset = SingleSetenceDataset(val_encodings, val_seq_labels, val_ner_labels)
    test_dataset = SingleSetenceDataset(test_encodings, test_seq_labels, test_ner_labels)

    return train_dataset, test_dataset, val_dataset, len(unique_seq_labels), torch.ones([len(unique_seq_labels)])
    



def _read_data_dialogue_action(file_path):
    if os.path.exists(os.path.join(file_path, "test.json")):
        test_file_path = file_path
    else:
        test_file_path = "/".join(file_path.rstrip("/").split("/")[:-2])
    
    assert os.path.exists(os.path.join(test_file_path, "test.json")), "No test file at {} or {}".format(file_path, test_file_path)
    
    train_data = json.load(open(os.path.join(file_path, "train.json"), "r"))
    test_data = json.load(open(os.path.join(test_file_path, "test.json"), "r"))
    val_data = json.load(open(os.path.join(file_path, "dev.json"), "r"))

    train_text, train_labels = train_data["text"], train_data['label']
    test_text, test_labels = test_data["text"], test_data['label']
    val_text, val_labels = val_data["text"], val_data['label']

    return train_text, train_labels, test_text, test_labels, val_text, val_labels


def _read_data_response_selection(file_path):
    if os.path.exists(os.path.join(file_path, "test.txt")):
        test_file_path = file_path
    else:
        test_file_path = "/".join(file_path.rstrip("/").split("/")[:-2])
    
    assert os.path.exists(os.path.join(test_file_path, "test.txt")), "No test file at {} or {}".format(file_path, test_file_path)
    
    train_data = list(csv.reader((open(os.path.join(file_path, "train.txt"), "r")), delimiter="\t"))
    test_data = list(csv.reader((open(os.path.join(test_file_path, "test.txt"), "r")), delimiter="\t"))
    val_data = list(csv.reader((open(os.path.join(file_path, "dev.txt"), "r")), delimiter="\t"))


    train_context = [t[0] for t in train_data]
    test_context = [t[0] for t in test_data]
    val_context = [t[0] for t in val_data]

    train_response = [t[1] for t in train_data]
    test_response = [t[1] for t in test_data]
    val_response = [t[1] for t in val_data]


    return train_context, train_response, test_context, test_response, val_context, val_response 





def _tokenize_multiturn_dialogue_concatenate(raw_text, tokenizer, max_seq_len, change_usrsys_to_sep_token=False, data_clean=False):
    if data_clean:    
        cleaned_text = [t.replace("[SEP]", "") for t in raw_text]
        if change_usrsys_to_sep_token:
            cleaned_text = [t.replace("[USR]", "[SEP]").replace("[SYS]", "[SEP]") for t in raw_text]
        num_keeped_words = int(max_seq_len*0.9)
        cleaned_text = [" ".join(t.split()[-num_keeped_words:]) for t in cleaned_text]
    else:
        cleaned_text = raw_text
    
    print(len(cleaned_text))
    print(cleaned_text[0])


    token_feat = tokenizer.batch_encode_plus(
        cleaned_text, 
        max_length=max_seq_len, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )

    
    return token_feat
        



def get_dialogue_action_dataset(BERT_MODEL="bert-base-uncased", file_path="", max_seq_length=32, concatenate=False, num_turn=2):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    change_usrsys_to_sep_token = True if "todbert" not in BERT_MODEL.lower() else False


    train_text, train_labels, test_text, test_labels, val_text, val_labels = _read_data_dialogue_action(file_path)
    
    print("Number of training samples: ", len(train_text))

    train_encodings = _tokenize_multiturn_dialogue_concatenate(train_text, tokenizer, max_seq_len=max_seq_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=True)
    val_encodings = _tokenize_multiturn_dialogue_concatenate(val_text, tokenizer, max_seq_len=max_seq_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=True)
    test_encodings = _tokenize_multiturn_dialogue_concatenate(test_text, tokenizer, max_seq_len=max_seq_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=True)

    train_dataset = SingleSetenceDataset(train_encodings, train_labels)
    val_dataset = SingleSetenceDataset(val_encodings, val_labels)
    test_dataset = SingleSetenceDataset(test_encodings, test_labels)

    unique_seq_labels = len(train_labels[0])
    print("Number of actions: ", unique_seq_labels)

    return train_dataset, test_dataset, val_dataset, unique_seq_labels



def get_response_selection_dataset(BERT_MODEL="bert-base-uncased", file_path="", max_seq_length=32, concatenate=False, num_turn=2):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    change_usrsys_to_sep_token = True if "tod" not in BERT_MODEL.lower() else False
    # change_usrsys_to_sep_token = True

    max_query_length = max_seq_length
    max_response_length = max_seq_length if 'amazonqa' in file_path else 32
    data_clean = False if 'amazonqa' in file_path else True

    train_context, train_response, test_context, test_response, val_context, val_response = _read_data_response_selection(file_path)
    
    print("Number of training samples: ", len(train_context))

    train_context_encodings = _tokenize_multiturn_dialogue_concatenate(train_context, tokenizer, max_seq_len=max_query_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)
    val_context_encodings = _tokenize_multiturn_dialogue_concatenate(val_context, tokenizer, max_seq_len=max_query_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)
    test_context_encodings = _tokenize_multiturn_dialogue_concatenate(test_context, tokenizer, max_seq_len=max_query_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)

    train_response_encodings = _tokenize_multiturn_dialogue_concatenate(train_response, tokenizer, max_seq_len=max_response_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)
    val_response_encodings = _tokenize_multiturn_dialogue_concatenate(val_response, tokenizer, max_seq_len=max_response_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)
    test_response_encodings = _tokenize_multiturn_dialogue_concatenate(test_response, tokenizer, max_seq_len=max_response_length, change_usrsys_to_sep_token=change_usrsys_to_sep_token, data_clean=data_clean)


    train_dataset = SetencePairDataset(train_context_encodings, train_response_encodings)
    val_dataset = SetencePairDataset(val_context_encodings, val_response_encodings)
    test_dataset = SetencePairDataset(test_context_encodings, test_response_encodings)

    return train_dataset, test_dataset, val_dataset



def _sample_few_shot_from_original_data(TASK="seq", data_ratio=1, train_text=None, train_seq_labels=None, train_ner=None, seed=24):
    random.seed(seed)
    num_shot = int(data_ratio) if data_ratio > 1 else max(1, int(data_ratio*len(train_seq_labels)))
    Label2id = {}
    if TASK == "seq":
        Label2id = {}
        for i, seq_label in enumerate(train_seq_labels):
            Label2id[seq_label] = Label2id.get(seq_label, []) + [i]
    elif TASK == "da":
        for i, seq_label in enumerate(train_seq_labels):
            copy_seq_label = np.array(deepcopy(seq_label))
            for act in np.where(copy_seq_label==1)[0]:
                Label2id[act] = Label2id.get(act, []) + [i]
    elif TASK == "ner":
        for i, ner_label in enumerate(train_ner):
            current_ner_label = set(ner_label)
            for l in current_ner_label:
                Label2id[l] = Label2id.get(l, []) + [i]
    
    train_ner_new = []
    train_text_new = []
    train_seq_labels_new = []
    val_ner_new = []
    val_text_new = []
    val_seq_labels_new = []


    all_sampled_ids = {}
    for label in Label2id:
        num_sample = min(num_shot*2, len(Label2id[label]))
        cur_sampled_ids = random.sample(Label2id[label], num_sample)
        for cur_id in cur_sampled_ids:
            all_sampled_ids[label] = all_sampled_ids.get(label, []) + [cur_id]
    
    unique_samples = set()
    for label in all_sampled_ids:
        cur_samples = all_sampled_ids[label]
        cur_samples = [s for s in cur_samples if s not in unique_samples]
        unique_samples = unique_samples.union(set(cur_samples))

        num_train_samples = len(cur_samples) - int(len(cur_samples) / 2)
        train_samples = cur_samples[:num_train_samples]
        val_samples = cur_samples[num_train_samples:]

        for sample in train_samples:
            if TASK == "ner":
                train_ner_new.append(train_ner[sample])
            train_text_new.append(train_text[sample])
            train_seq_labels_new.append(train_seq_labels[sample])

        for sample in val_samples:
            if TASK == "ner":
                val_ner_new.append(train_ner[sample])
            val_text_new.append(train_text[sample])
            val_seq_labels_new.append(train_seq_labels[sample])

    return train_ner_new, train_text_new, train_seq_labels_new, val_ner_new, val_text_new, val_seq_labels_new





'''
### example usage of _generate_few_shot_data_files
for data in ['wmoz', 'dstc2', 'sim_joint']:
    for data_ratio in [10, 20, 50]:
        _generate_few_shot_data_files(data_dir="/mnt/efs/data/da/" + data, data_ratio=data_ratio, num_runs=10, task='da')
'''

def _generate_few_shot_data_files(data_dir, data_ratio, num_runs, task="seq"):
    # for few-shot setting
    if task in ['seq', 'ner', 'oos']:
        train_ner_all, test_ner, train_text_all, test_text, train_seq_labels_all, test_seq_labels = _read_data_intent_slot(data_dir)
        for seed in range(num_runs):
            root_path = os.path.join(data_dir, str(int(data_ratio)))
            root_path = os.path.join(root_path, str(seed))
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            train_ner, train_text, train_seq_labels, val_ner, val_text, val_seq_labels =  _sample_few_shot_from_original_data("seq", data_ratio, train_text_all, train_seq_labels_all, train_ner_all, seed)

            with open(os.path.join(root_path, "seq_train.txt"), "w") as f:
                for i in range(len(train_text)):
                    f.write(train_text[i] + "\t" + str(train_seq_labels[i]) + "\n")
            with open(os.path.join(root_path, "seq_val.txt"), "w") as f:
                for i in range(len(val_text)):
                    f.write(val_text[i] + "\t" + str(val_seq_labels[i]) + "\n")

            if task == 'ner':
                with open(os.path.join(root_path, "ner_train.txt"), "w") as f:
                    for i in range(len(train_text)):
                        f.write(" ".join(train_ner[i]) + "\n")
                with open(os.path.join(root_path, "ner_val.txt"), "w") as f:
                    for i in range(len(val_text)):
                        f.write(" ".join(val_ner[i]) + "\n")
                with open(os.path.join(root_path, "ner_test.txt"), "w") as f:
                    for i in range(len(test_text)):
                        f.write(" ".join(test_ner[i]) + "\n")
    

    elif task == "da":
        train_text_all, train_seq_labels_all, _, _, _, _ = _read_data_dialogue_action(data_dir)

        for seed in range(num_runs):
            root_path = os.path.join(data_dir, str(int(data_ratio)))
            root_path = os.path.join(root_path, str(seed))
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            
            _, train_text, train_seq_labels, _, val_text, val_seq_labels = _sample_few_shot_from_original_data("da", data_ratio, train_text_all, train_seq_labels_all, None, seed)
            train_data = {"text": train_text, "label": train_seq_labels}
            val_data = {"text": val_text, "label": val_seq_labels}

            with open(os.path.join(root_path, "train.json"), "w") as f:
                json.dump(train_data, f)
            
            with open(os.path.join(root_path, "dev.json"), "w") as f:
                json.dump(val_data, f)

    elif task == "rs":
        train_context_all, train_response_all, _, _, val_context_all, val_response_all = _read_data_response_selection(data_dir)
        
        num_train_all = len(train_context_all)
        num_dev_all = len(val_context_all)

        num_train_1_percent = int(num_train_all * 0.01)
        num_train_3_percent = int(num_train_all * 0.03)

        num_dev_1_percent = max(num_train_1_percent // 100, 1) * 100
        num_dev_3_percent = max(num_train_3_percent // 100, 1) * 100

        print(num_train_1_percent)
        print(num_train_3_percent)
        print(num_dev_1_percent)
        print(num_dev_3_percent)
        
        for seed in range(num_runs):
            random.seed(seed)
            root_path_1 = os.path.join(data_dir, str(1))
            root_path_1 = os.path.join(root_path_1, str(seed))
            if not os.path.exists(root_path_1):
                os.makedirs(root_path_1)

            root_path_3 = os.path.join(data_dir, str(3))
            root_path_3 = os.path.join(root_path_3, str(seed))
            if not os.path.exists(root_path_3):
                os.makedirs(root_path_3)

            train_1_percent_idx = np.array(random.sample(list(range(num_train_all)), num_train_1_percent))
            train_3_percent_idx = np.array(random.sample(list(range(num_train_all)), num_train_3_percent))
            dev_1_percent_idx = np.array(random.sample(list(range(num_dev_all)), num_dev_1_percent))
            dev_3_percent_idx = np.array(random.sample(list(range(num_dev_all)), num_dev_3_percent))

            with open(os.path.join(root_path_1, "train.txt"), "w") as f:
                for idx in train_1_percent_idx:
                    f.write(train_context_all[idx] + "\t" + train_response_all[idx] + "\n")
            
            with open(os.path.join(root_path_3, "train.txt"), "w") as f:
                for idx in train_3_percent_idx:
                    f.write(train_context_all[idx] + "\t" + train_response_all[idx] + "\n")
            
            with open(os.path.join(root_path_1, "dev.txt"), "w") as f:
                for idx in dev_1_percent_idx:
                    f.write(val_context_all[idx] + "\t" + val_response_all[idx] + "\n")
            
            with open(os.path.join(root_path_3, "dev.txt"), "w") as f:
                for idx in dev_3_percent_idx:
                    f.write(val_context_all[idx] + "\t" + val_response_all[idx] + "\n")
    

    else:
        raise ValueError()


