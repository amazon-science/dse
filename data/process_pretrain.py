import os
import pickle
import random
import argparse
import csv
from copy import deepcopy

random.seed(24)

def generate_single_vs_single(data_dir, output_dir, min_length):
    with open(data_dir, "rb") as f:
        data = pickle.load(f)

    dialogues = []
    for dataset in data:
        for split in ['train', 'dev', 'test']:
            if len(data[dataset][split]) > 0:
                for item in data[dataset][split]:
                    cur_dia = deepcopy(item['dialog_history'][1:])
                    if item['turn_sys'] not in cur_dia:
                        cur_dia.append(item['turn_sys'])
                        cur_dia.append(item['turn_usr'])
                    cur_dia = [t.replace("\n", "").replace("\t", "") for t in cur_dia]
                    # get positive examples
                    for i in range(len(cur_dia) - 1):
                        dialogues.append([cur_dia[i], cur_dia[i+1], 1])


    dialogues = [d for d in dialogues if len(d[0].split()) > min_length and len(d[1].split()) > min_length]

    random.shuffle(dialogues)

    root_dir = "/".join(output_dir.split("/")[:-1])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    f = open(output_dir, "w")
    d_w = csv.writer(f, delimiter='\t')
    for dia in dialogues:
        d_w.writerow([dia[0], dia[1]])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='directory to the pickle file')
    parser.add_argument('--output_dir', type=str, required=True, help="directory to the output file")
    parser.add_argument('--min_length', type=int, default=3, help="sentences whose length are less or equal to this number are filtered out")

    args = parser.parse_args()
    generate_single_vs_single(args.data_dir, args.output_dir, args.min_length)