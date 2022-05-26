import sys
sys.path.append("../evaluate")

import os
import csv
import json
import random
import argparse
from datasets import load_dataset
from utils.data import _generate_few_shot_data_files



def process_clinc150(output_dir):
    clinc150 = load_dataset("clinc_oos", "plus")
    train = []
    test = []
    valid = []

    for item in clinc150["train"]:
        train.append([item['text'], item['intent']])
    for item in clinc150["test"]:
        test.append([item['text'], item['intent']])
    for item in clinc150["validation"]:
        valid.append([item['text'], item['intent']])

    def change_label(data):
        for item in data:
            if item[1] == 42:
                item[1] = 150
            elif item[1] == 150:
                item[1] = 42
        return data

    train = change_label(train)
    test = change_label(test)
    valid = change_label(valid)

    train_nooos = [item for item in train if item[1] < 150]
    test_nooos = [item for item in test if item[1] < 150]
    valid_nooos = [item for item in valid if item[1] < 150]

    clinc150_path = os.path.join(output_dir, "intent", "clinc150")
    clinc150_all_path = os.path.join(output_dir, "intent", "clinc150_all")
    os.makedirs(clinc150_path)
    os.makedirs(clinc150_all_path)
    with open(os.path.join(clinc150_path, "seq_train.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(train_nooos)
    with open(os.path.join(clinc150_path, "seq_test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(test_nooos)
    with open(os.path.join(clinc150_path, "seq_valid.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(valid_nooos)
    with open(os.path.join(clinc150_all_path, "seq_train.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(train)
    with open(os.path.join(clinc150_all_path, "seq_test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(test)
    with open(os.path.join(clinc150_all_path, "seq_valid.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(valid)

    # generate few shot samples
    for path in [clinc150_path, clinc150_all_path]:
        for data_ratio in [1.1, 5]:
            _generate_few_shot_data_files(data_dir=path, data_ratio=data_ratio, num_runs=10, task="seq")


def process_bank77(output_dir):
    banking77 = load_dataset("banking77")
    train = []
    test = []

    for item in banking77["train"]:
        train.append([item['text'], item['label']])
    for item in banking77["test"]:
        test.append([item['text'], item['label']])

    bank77_path = os.path.join(output_dir, "intent", "bank77")
    os.makedirs(bank77_path)
    with open(os.path.join(bank77_path, "seq_train.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(train)
    with open(os.path.join(bank77_path, "seq_test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(test)

    # generate few shot samples
    for data_ratio in [1.1, 5]:
        _generate_few_shot_data_files(data_dir=bank77_path, data_ratio=data_ratio, num_runs=10, task="seq")


def process_snips(output_dir):
    assert os.path.exists("snips_train.json"), "please download the snips_train.json and snips_test.json as instructed and put it in current directory"
    with open("snips_train.json", "r") as f:
        snips_train = json.load(f)
    with open("snips_test.json", "r") as f:
        snips_test = json.load(f)

    snips = snips_train + snips_test
    cat2id = {}
    idx = 0
    for item in snips:
        if item[1] not in cat2id:
            cat2id[item[1]] = idx
            idx += 1
        item[1] = cat2id[item[1]]

    # randomly select 90% of data for training and the rest for testing
    total_count = {}
    for i in range(7):
        total_count[i] = 0
    for item in snips:
        total_count[item[1]] += 1

    cat_count = {}
    for i in range(7):
        cat_count[i] = 0

    train = []
    test = []
    for item in snips:
        cat = item[1]
        if cat_count[cat] >= total_count[cat] * 0.9:
            test.append(item)
            cat_count[cat] += 1
        else:
            train.append(item)
            cat_count[cat] += 1

    snips_path = os.path.join(output_dir, "intent", "snips")
    os.makedirs(snips_path)
    with open(os.path.join(snips_path, "seq_train.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(train)
    with open(os.path.join(snips_path, "seq_test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(test)

    # generate few shot samples
    for data_ratio in [1.1, 5]:
        _generate_few_shot_data_files(data_dir=snips_path, data_ratio=data_ratio, num_runs=10, task="seq")



def process_hwu64(output_dir):
    assert os.path.exists("NLU-Data-Home-Domain-Annotated-All.csv"), "please download NLU-Data-Home-Domain-Annotated-All.csv as instructed and put it in current directory"
    with open("NLU-Data-Home-Domain-Annotated-All.csv", "r") as f:
        raw_data = list(csv.reader(f, delimiter=";"))[1:]

    with open("hwu64_label2id.json", "r") as f:
        label2id = json.load(f)

    hwu64 = []
    for item in raw_data:
        raw_label = item[2] + "_" + item[3]
        if raw_label not in label2id:
            continue
        label = label2id[raw_label]
        text = item[9]
        hwu64.append([text, label])

    # randomly select 85% of data for training and the rest for testing
    total_count = {}
    for i in range(64):
        total_count[i] = 0
    for item in hwu64:
        total_count[item[1]] += 1

    cat_count = {}
    for i in range(64):
        cat_count[i] = 0

    train = []
    test = []
    for item in hwu64:
        cat = item[1]
        if cat_count[cat] >= total_count[cat] * 0.85:
            test.append(item)
            cat_count[cat] += 1
        else:
            train.append(item)
            cat_count[cat] += 1

    hwu64_path = os.path.join(output_dir, "intent", "hwu64")
    os.makedirs(hwu64_path)
    with open(os.path.join(hwu64_path, "seq_train.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(train)
    with open(os.path.join(hwu64_path, "seq_test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(test)

    # generate few shot samples
    for data_ratio in [1.1, 5]:
        _generate_few_shot_data_files(data_dir=hwu64_path, data_ratio=data_ratio, num_runs=10, task="seq")



def process_amazonqa(output_dir):
    assert os.path.exists("amazonqa.txt"), "please use ParlAI to download amazonqa.txt as instructed and put it in current directory"
    with open("amazonqa.txt", "r") as f:
        amazonqa = list(csv.reader(f, delimiter="\t"))

    all_qa = []
    for item in amazonqa:
        try:
            if len(item) > 0:
                all_qa.append([item[0][5:], item[1][7:]])
        except:
            continue

    amazonqa_path = os.path.join(output_dir, "rs", "amazonqa")
    os.makedirs(amazonqa_path)
    with open(os.path.join(amazonqa_path, "test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(all_qa)


def process_ubuntu(output_dir):
    assert os.path.exists("ubuntu_test.txt"), "please use ParlAI to download ubuntu_test.txt and ubuntu_valid.txt as instructed and put it in current directory"
    with open("ubuntu_test.txt", "r") as f:
        ubuntu_test = list(csv.reader(f, delimiter="\t"))
    with open("ubuntu_valid.txt", "r") as f:
        ubuntu_val = list(csv.reader(f, delimiter="\t"))

    ubuntu = ubuntu_test + ubuntu_val
    ubuntu = [u for u in ubuntu if len(u) > 0]
    data = []
    for item in ubuntu:
        data.append([item[0][5:].replace("\\n", " "), item[1][7:].replace("\\n", " "), item[2][17:].replace("\\n", " ")])

    ubuntu_path = os.path.join(output_dir, "rs", "ubuntu")
    os.makedirs(ubuntu_path)
    with open(os.path.join(ubuntu_path, "test.txt"), "w") as f:
        f_w = csv.writer(f, delimiter="\t")
        f_w.writerows(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="directory to the output file")
    parser.add_argument('--task', type=str, required=True, help="choose from [bank77, clinc150, hwu64, snips, amazonqa, ubuntu]")


    args = parser.parse_args()
    if args.task == "clinc150":
        process_clinc150(args.output_dir)
    elif args.task == "bank77":
        process_bank77(args.output_dir)
    elif args.task == "snips":
        process_snips(args.output_dir)
    elif args.task == "hwu64":
        process_hwu64(args.output_dir)
    elif args.task == "amazonqa":
        process_amazonqa(args.output_dir)
    elif args.task == "ubuntu":
        process_ubuntu(args.output_dir)
    else:
        raise ValueError("choose task from [bank77, clinc150, hwu64, snips, amazonqa, ubuntu]")