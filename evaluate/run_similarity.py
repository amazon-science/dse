import os
import csv
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from utils.data import get_intent_slot_dataset, _read_data_intent_slot
from utils.get_embeddings import calculate_embedding
from utils.test_util import test_oos, test_response_selection, test_intent_retrieval


# fill empty sentence '' with a '.' to avoid empty input to the model
def fill_empty_sentence(texts):
    for i, t in enumerate(texts):
        if t.strip() == "":
            texts[i] = "."


'''
This function load pre-trained model from 'model_dir', perform intent classification with prototypical network and 
record all the experimental results at 'output_dir'
'''
def eval_intent_classification_with_prototypical_net(model, tokenizer, model_dir, data_root_dir, output_dir='results/intent', task_type='average_embedding', num_runs=10, max_seq_length=50):
    print("\n\n Evaluating intent classification")

    datasets = ['appen_asr', 'appen_human', 'clinc150', 'bank77', 'snips', 'hwu64']

    for dataset in datasets:
        for data_ratio in [1, 5]:
            print(f"Making prediction for {dataset} with {data_ratio}-shot data")
            data_dir = os.path.join(data_root_dir, "intent", dataset)

            hidden_size = model.config.hidden_size

            accuracy_count = 0


            for idx in range(num_runs):
                DATA_DIR = os.path.join(data_dir, str(int(data_ratio)))
                DATA_DIR = os.path.join(DATA_DIR, str(idx))
                _, _, train_text, test_text, support_labels, query_labels = _read_data_intent_slot(DATA_DIR, valid=False, ner=False)

                support_embedding = calculate_embedding(train_text, model, tokenizer, max_length=max_seq_length, task_type=task_type)
                support_embedding = normalize(support_embedding)
                num_category = len(set(support_labels))
                support_labels = np.array(support_labels)
                prototype_embedding = np.zeros([num_category, support_embedding.shape[1]])
                # Use the average embedding of all the samples that belongs to this category as the category's prototype embedding   
                for cat in range(num_category):
                    cat_embedding = support_embedding[np.where(support_labels==cat)[0]].mean(axis=0) if len(np.where(support_labels==cat)[0]) else np.zeros([hidden_size])
                    prototype_embedding[cat] = cat_embedding

                query_embedding = calculate_embedding(test_text, model, tokenizer, max_length=max_seq_length, task_type=task_type)
                query_embedding = normalize(query_embedding)

                sim_matrix = query_embedding @ prototype_embedding.T
                preds = sim_matrix.argmax(axis=1)
                labels = np.array(query_labels)
                acc = 100. * len(np.where(preds==labels)[0]) / len(preds)

                print("Current accuracy: ", acc)
                accuracy_count += acc

            prefix = dataset + "|" + str(int(data_ratio)) + "|" + model_dir.split("/")[-1]  + "|" + task_type
            average_acc = accuracy_count/num_runs
            print("Average accuracy: \n\n", average_acc)
            with open(os.path.join(output_dir, "results.txt"), "a") as f:
                f.write(prefix + " average_acc: "  + str(average_acc) + "\n")


'''
This function load pre-trained model from 'model_dir', perform utterance retrieval on intent classification datasets and 
record all the experimental results at 'output_dir'
'''
def eval_utterance_retrieval(model, tokenizer, model_dir, data_root_dir, output_dir='results/utterance_retrieval', task_type='average_embedding', max_seq_length=50):
    print("\n\n Evaluating utterance retrieval")
    
    datasets = ['clinc150', 'bank77', 'snips', 'hwu64', 'atis', 'appen_asr', 'appen_human']

    for dataset in datasets:
        print(f"Making prediction for {dataset}")
        text_dir = os.path.join(data_root_dir, "intent", dataset, 'seq_train.txt')

        with open(text_dir, "r") as f:
            data = f.readlines()
            data = [s.strip("\n").split("\t") for s in data]

        all_sents = [d[0] for d in data]
        all_labels = [d[1] for d in data]


        embeddings = calculate_embedding(all_sents, model, tokenizer, max_length=max_seq_length, task_type=task_type)
        embeddings = normalize(embeddings)


        count = test_intent_retrieval(all_labels, embeddings)
        message = "|".join([dataset, str(count[5]), str(count[10]), str(count[20]), str(count[50]), str(count[100])])
        print(message)
        with open(os.path.join(output_dir, "results.txt"), "a") as f:
            f.write(message + "\n")


'''
This function load pre-trained model from 'model_dir', perform response selection and 
record all the experimental results at 'output_dir'
'''
def eval_response_selection_amazonqa(model, tokenizer, model_dir, data_root_dir, output_dir='results/rs', task_type='average_embedding', max_seq_length=128):
    print("\n\n Evaluating response selection on AmazonQA")
    
    data_dir = os.path.join(data_root_dir, 'rs', 'amazonqa', 'test.txt')
    with open(data_dir, "r") as f:
        data = list(csv.reader(f, delimiter="\t"))
        query = [d[0] for d in data] 
        response = [d[1] for d in data] 

    fill_empty_sentence(query)
    fill_empty_sentence(response)


    q_embeddings = calculate_embedding(query, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=1000, verbose=True)
    r_embeddings = calculate_embedding(response, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=1000, verbose=True)

    q_embeddings = normalize(q_embeddings)
    r_embeddings = normalize(r_embeddings)

    num_batches = int(q_embeddings.shape[0] / 100)
    recall_1, recall_3, recall_5, recall_10 = 0, 0, 0, 0
    for i in range(num_batches):
        cur_recall_1, cur_recall_3, cur_recall_5, cur_recall_10 = test_response_selection(q_embeddings[i*100:(i+1)*100], r_embeddings[i*100:(i+1)*100])
        recall_1 += cur_recall_1
        recall_3 += cur_recall_3
        recall_5 += cur_recall_5
        recall_10 += cur_recall_10

    recall_1 /= num_batches
    recall_3 /= num_batches
    recall_5 /= num_batches
    recall_10 /= num_batches

    message = str(recall_1) + "|" + str(recall_3) + "|" + str(recall_5) + "|" + str(recall_10)
    print(message)
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(message + "\n")


def eval_response_selection_ubuntu(model, tokenizer, model_dir, data_root_dir, output_dir='results/rs', task_type='average_embedding', max_seq_length=256):
    print("\n\n Evaluating response selection on Ubuntu")
    
    data_dir = os.path.join(data_root_dir, 'rs', 'ubuntu', 'test.txt')
    with open(data_dir, "r") as f:
        data = list(csv.reader(f, delimiter="\t"))
        query = [d[0] for d in data] 
        target = [d[1] for d in data] 
        candidate = [d[2] for d in data]
    
    target_plus_candidate = []
    for t, c in zip(target, candidate):
        c = c.split("|")
        target_plus_candidate.append(t)
        target_plus_candidate.extend(c)
    
    fill_empty_sentence(query)
    fill_empty_sentence(target_plus_candidate)

    num_sampels = len(data)

    q_embeddings = calculate_embedding(query, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=1000, verbose=True)
    t_c_embeddings = calculate_embedding(target_plus_candidate, model, tokenizer, max_length=32, task_type=task_type, batch_size=2000, verbose=True)

    q_embeddings = normalize(q_embeddings)
    t_c_embeddings = normalize(t_c_embeddings)
    all_embeddings = t_c_embeddings.reshape([num_sampels, 101, -1])

    all_similarity = (q_embeddings.reshape([num_sampels, 1, -1]) * all_embeddings).sum(axis=2)
    predictions = np.argsort(all_similarity, axis=1)


    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    recall_10 = 0

    for pred in predictions:
        recall_1 += int(0 in pred[-1:])
        recall_3 += int(0 in pred[-3:])
        recall_5 += int(0 in pred[-5:])
        recall_10 += int(0 in pred[-10:])

    recall_1 /= num_sampels / 100.
    recall_3 /= num_sampels / 100.
    recall_5 /= num_sampels / 100.
    recall_10 /= num_sampels / 100.

    message = str(recall_1) + "|" + str(recall_3) + "|" + str(recall_5) + "|" + str(recall_10)
    print(message)
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(message + "\n")






def eval_natural_language_inference(model, tokenizer, model_dir, data_root_dir, output_dir='results/nli_sim', task_type='average_embedding', max_seq_length=50):
    print("\n\n Evaluating natural language inference on dialog-nli using {}".format(task_type))
    
    data_dir = os.path.join(data_root_dir, 'nli', 'nli', 'all.txt')
    with open(data_dir, "r") as f:
        data = list(csv.reader(f, delimiter="\t"))
        query = [d[0] for d in data] 
        pos = [d[1] for d in data] 
        neg = [d[2] for d in data] 


    
    query_embeddings = calculate_embedding(query, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=2000, verbose=True)
    neg_embeddings = calculate_embedding(neg, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=2000, verbose=True)
    pos_embeddings = calculate_embedding(pos, model, tokenizer, max_length=max_seq_length, task_type=task_type, batch_size=2000, verbose=True)

    query_embeddings = normalize(query_embeddings)
    neg_embeddings = normalize(neg_embeddings)
    pos_embeddings = normalize(pos_embeddings)

    neg_scores = (query_embeddings * neg_embeddings).sum(axis=1)
    pos_scores = (query_embeddings * pos_embeddings).sum(axis=1)
    print(neg_scores)
    print(pos_scores)

    num_correct = (pos_scores > neg_scores).sum()
    accuracy = 100. * num_correct / len(data)

    message = str(accuracy) 
    print(message)
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(message + "\n")





'''
Load pre-trained model from 'model_dir', perform out of scope and record all the experimental results at 'output_dir'
'''
def eval_out_of_scope_detection(model, tokenizer, model_dir, data_root_dir, output_dir='results/oos', task_type='average_embedding', num_runs=10, max_seq_length=50):
    print("\n\n Evaluating out-of-scope detection")
    
    for data_ratio in [1, 5]:    
        print(f"Making prediction with {data_ratio}-shot data")
        data_dir = os.path.join(data_root_dir, "intent", "clinc150_all")
        hidden_size = model.config.hidden_size

        count = {}
        for i in range(3):
            count[i] = {}
            count[i]['accuracy_count'] = 0
            count[i]['in_accuracy_count'] = 0
            count[i]['oos_accuracy_count'] = 0
            count[i]['oos_recall_count'] = 0


        for idx in range(num_runs):
            DATA_DIR = os.path.join(data_dir, str(int(data_ratio)))
            DATA_DIR = os.path.join(DATA_DIR, str(idx))
            _, _, train_text, test_text, support_labels, query_labels = _read_data_intent_slot(DATA_DIR, valid=False, ner=False)

            support_embedding = calculate_embedding(train_text, model, tokenizer, max_length=max_seq_length, task_type=task_type)
            support_embedding = normalize(support_embedding)
            num_category = len(set(support_labels))
            support_labels = np.array(support_labels)
            prototype_embedding = np.zeros([num_category, support_embedding.shape[1]])
            for cat in range(num_category):
                cat_embedding = support_embedding[np.where(support_labels==cat)[0]].mean(axis=0) if len(np.where(support_labels==cat)[0]) else np.zeros([hidden_size])
                prototype_embedding[cat] = cat_embedding

            query_embedding = calculate_embedding(test_text, model, tokenizer, max_length=max_seq_length, task_type=task_type)
            query_embedding = normalize(query_embedding)

            sim_matrix = query_embedding @ prototype_embedding.T
            sim_matrix = normalize(sim_matrix)

            labels = np.array(query_labels)

            max_values = sim_matrix.max(axis=1)
            cur_mean = max_values.mean()
            cur_std = max_values.std()
            
            thresholds = [cur_mean - cur_std, cur_mean, cur_mean + cur_std]
            for j in range(3):
                cur_threshold = thresholds[j]
                preds = np.zeros([len(sim_matrix)])
                for i, sim in enumerate(sim_matrix):
                    preds[i] = sim.argmax() if sim.max() > cur_threshold else 150

                acc, ins_acc, oos_acc, oos_recall = test_oos(labels, preds, oos_idx=150)
                count[j]['accuracy_count'] += acc
                count[j]['in_accuracy_count'] += ins_acc
                count[j]['oos_accuracy_count'] += oos_acc
                count[j]['oos_recall_count'] += oos_recall


        prefix = str(int(data_ratio)) + "|" 
        for i in range(3):
            average_acc = count[i]['accuracy_count']/num_runs
            average_in_acc = count[i]['in_accuracy_count']/num_runs
            average_oos_acc = count[i]['oos_accuracy_count']/num_runs
            average_oos_recall = count[i]['oos_recall_count']/num_runs

            print("Average accuracy: ", average_acc)
            print("Average in domain accuracy: ", average_in_acc)
            print("Average out domain accuracy: ", average_oos_acc)
            print("Average out domain recall: ", average_oos_recall)
            message = "|".join([str(i), str(average_acc), str(average_in_acc), str(average_oos_acc), str(average_oos_recall)])
            with open(os.path.join(output_dir, "results.txt"), "a") as f:
                f.write(message + "\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='directory to the pre-trained model')
    parser.add_argument('--data_root_dir', type=str, required=True, help="directory to the folder that contains all the evaluation data")
    parser.add_argument('--output_dir', type=str, required=True, help="directory to save all the evaluation results")
    parser.add_argument('--TASK', type=str, required=True, help="The task to perform. Choose from ['rs', 'intent', 'utterance_retrieval', 'oos']")
    parser.add_argument('--embedding_type', type=str, default='average_embedding', help="How to calculate sentence embedding. Choose from ['average_embedding', 'cls_embedding']")
    parser.add_argument('--num_runs', type=int, default=10, help="Number of independent experiments to run for 'oos' and 'intent'. ")
    parser.add_argument('--max_seq_length', type=int, default=50, help="Max length of the input sequence")


    args = parser.parse_args()    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)        

    if "simcse" in args.model_dir.lower():
        args.embedding_type = "cls_embedding"
    elif "dialogpt" in args.model_dir.lower():
        args.embedding_type = "dialogpt_embedding"

    args.model_dir = args.model_dir.replace("//","/")
    model = AutoModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.TASK == "intent":
        eval_intent_classification_with_prototypical_net(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.num_runs, args.max_seq_length)
    elif args.TASK == "oos":
        eval_out_of_scope_detection(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.num_runs, args.max_seq_length)
    elif args.TASK == "rs_ubuntu":
        eval_response_selection_ubuntu(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.max_seq_length)
    elif args.TASK == "rs_amazon":
        eval_response_selection_amazonqa(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.max_seq_length)
    elif args.TASK == "utterance_retrieval":
        eval_utterance_retrieval(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.max_seq_length)
    elif args.TASK == "nli":
        eval_natural_language_inference(model, tokenizer, args.model_dir, args.data_root_dir, args.output_dir, args.embedding_type, args.max_seq_length)
    else:
        raise ValueError("Choose TASK from ['rs', 'intent', 'utterance_retrieval', 'oos']")
    
