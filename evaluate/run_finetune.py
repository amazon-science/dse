from tqdm import tqdm, trange
import logging
import argparse
import os
import pickle
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

from transformers import BertConfig, AutoTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from utils.data import get_intent_slot_dataset, get_dialogue_action_dataset, \
                 get_response_selection_dataset, get_nli_dataset
from utils.model import BertForSequenceClassification, \
                  BertMultiTurnForDialogueActionPredictionConcat, \
                  BertMultiTurnForResponseSelectionConcat, \
                  BertForNaturalLanguageInference, \
                  DistilBertForSequenceClassification, \
                  DistilBertMultiTurnForDialogueActionPredictionConcat, \
                  DistilBertMultiTurnForResponseSelectionConcat, \
                  DistilBertForNaturalLanguageInference
from utils.test_util import test_oos, test_response_selection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


'''
This function evaluates the model on a specific task. It supports 5 tasks: 1. intent classification (seq), 
2. out-of-scope detection (oos), 3: response selection (rs), 4: dialogue action prediction (da), 5: language inference (nli)

Return:
    1. A dictionary with metric name as key and metric value as value (e.g., {'accuracy':0.92})
    2. Loss 
'''
def evaluate(test_dataset, model, args, prefix="Test"):
    test_sampler = SequentialSampler(test_dataset)
    eval_batch_size = args.per_gpu_batch_size * args.n_gpu * 3 if args.TASK != 'rs' else 100
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, sampler=test_sampler)


    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    model.eval()
    test_iterator = tqdm(test_dataloader, desc="Iteration") if len(test_dataloader)>100 else test_dataloader
    seq_correct = 0
    all_seq_preds = torch.zeros([0])
    all_seq_labels = torch.zeros([0])

    softmax = nn.Softmax(dim=2)

    loss = 0
    loss_func = nn.CrossEntropyLoss()

    rs_scores = {
        'recall_1': 0,
        'recall_3': 0,
        'recall_5': 0,
        'recall_10': 0,
    }

    with torch.no_grad():
        for batch in test_iterator:
            if args.TASK == 'seq':
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                seq_labels = batch['seq_labels']
                outputs = model(input_ids, attention_mask=attention_mask, labels=seq_labels)

                # for seq
                seq_pred = outputs[1].argmax(dim=1, keepdim=True).cpu().squeeze(1)
                seq_correct += sum(seq_pred==seq_labels.cpu()).item()

                loss += torch.mean(outputs[0]).item()
            
            elif args.TASK == 'nli':
                s1_input_ids = batch['context']['input_ids'].to(args.device)
                s1_attention_mask = batch['context']['attention_mask'].to(args.device)
                s2_input_ids = batch['response']['input_ids'].to(args.device)
                s2_attention_mask = batch['response']['attention_mask'].to(args.device)
                labels = batch['seq_labels'].to(args.device)
                outputs = model(s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask, labels=labels)
                seq_pred = outputs[1].argmax(dim=1, keepdim=True).cpu().squeeze(1)
                seq_correct += sum(seq_pred==labels.cpu()).item()

                loss += torch.mean(outputs[0]).item()


            elif args.TASK == 'oos':
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                seq_labels = batch['seq_labels']
                outputs = model(input_ids, attention_mask=attention_mask, labels=seq_labels)
                seq_pred = outputs[1].argmax(dim=1, keepdim=True).cpu().squeeze(1)

                all_seq_labels = torch.cat([all_seq_labels, seq_labels.cpu()])
                all_seq_preds = torch.cat([all_seq_preds, seq_pred])
                
                loss += torch.mean(outputs[0]).item()
            

            elif args.TASK == 'da':
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                seq_labels = batch['seq_labels']
                outputs = model(input_ids, attention_mask=attention_mask, labels=seq_labels)
                seq_pred = outputs[1].cpu()
                seq_pred = (seq_pred > 0.5).long()

                all_seq_labels = torch.cat([all_seq_labels, seq_labels.cpu()])
                all_seq_preds = torch.cat([all_seq_preds, seq_pred])

                loss += torch.mean(outputs[0]).item()
            

            elif args.TASK == "rs":
                context_input_ids = batch['context']['input_ids'].to(args.device)
                context_attention_mask = batch['context']['attention_mask'].to(args.device)
                response_input_ids = batch['response']['input_ids'].to(args.device)
                response_attention_mask = batch['response']['attention_mask'].to(args.device)
                context_output, response_output = model(context_input_ids, context_attention_mask, response_input_ids, response_attention_mask)

                logits = torch.mm(context_output, response_output.t().contiguous())
                batch_size = logits.shape[0]
                labels = torch.tensor(np.arange(batch_size)).to(logits.device)
                cur_loss = loss_func(logits, labels)

                recall_1, recall_3, recall_5, recall_10 = test_response_selection(context_output.detach().cpu().numpy(), response_output.detach().cpu().numpy())

                rs_scores['recall_1'] += recall_1
                rs_scores['recall_3'] += recall_3
                rs_scores['recall_5'] += recall_5
                rs_scores['recall_10'] += recall_10
                
                loss += torch.mean(cur_loss).item()
           

    loss /= len(test_dataloader)
    model.train()

    # logger.info('\n')
    if args.TASK in ['seq', 'nli']:
        accuracy = 100. * seq_correct / len(test_dataloader.dataset)
        message = 'Seq Accuracy: {}, Loss: {}'.format(accuracy, loss)
        main_metric = accuracy
    
    elif args.TASK == 'oos':
        acc, in_acc, oos_acc, oos_recall = test_oos(all_seq_labels.numpy(), all_seq_preds.numpy())
        message = 'Accuracy: %.4f, In Acc: %.4f, Out Acc: %.4f, Out Recall: %.4f, Loss: %.4f' % (acc, in_acc, oos_acc, oos_recall, loss)
    
    elif args.TASK == 'da':
        micro_f1 = f1_score(all_seq_labels, all_seq_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(all_seq_labels, all_seq_preds, average='macro', zero_division=0)
        message = 'Micro F1: %.4f, Macro F1: %.4f, Loss: %.4f' % (micro_f1, macro_f1, loss)

    elif args.TASK == 'rs':
        for metric in rs_scores:
            rs_scores[metric] /= len(test_dataloader)
        message = 'Recall@ 1: %.4f, 3: %.4f, 5: %.4f, 10: %.4f,' % (rs_scores['recall_1'], rs_scores['recall_3'], rs_scores['recall_5'], rs_scores['recall_10'])

    logger.info(prefix + " : " + message)

    if prefix == "Test":
        with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
            f.write(message + "\n")

    if args.TASK == 'oos':
        return {"main":acc, "in_acc":in_acc, "oos_acc":oos_acc, "oos_recall":oos_recall}, loss
    elif args.TASK == 'da':
        return {"main":(micro_f1 + macro_f1) / 2, "micro_f1":micro_f1, "macro_f1":macro_f1}, loss
    elif args.TASK == 'rs':
        main_metric = rs_scores['recall_1'] + rs_scores['recall_3'] + rs_scores['recall_5'] + rs_scores['recall_10']
        results = {'main':main_metric}
        results.update(rs_scores)
        return results, loss
    else:
        return {"main":main_metric}, loss


'''
This function train the model on a specific task. It supports 5 tasks: 1. intent classification (seq), 
2. out-of-scope detection (oos), 3: response selection (rs), 4: dialogue action prediction (da), 5: language inference (nli)

During the training process, the model is evaluated on the validation set every 'eval_step'. The model checkpoint which 
achieves the best performance on the validation set is saved for test.

Return:
    1. A dictionary with metric name as key and metric value as value (e.g., {'accuracy':0.92}). Represent the results
        on test set given by the checkpoint which achieves the best performance on validation set.

    2. The best checkpoint
'''
def train(args, model, train_dataset, val_dataset, test_dataset):
    crf_transitions = ['crf.transitions']
    crf_ratio = ['crf.ratio']
    crf_transitions_list = list(filter(lambda kv: kv[0] in crf_transitions, model.named_parameters()))
    crf_ratio_list = list(filter(lambda kv: kv[0] in crf_ratio, model.named_parameters()))
    bert_list = list(
        filter(lambda kv: kv[0] not in crf_ratio and kv[0] not in crf_transitions, model.named_parameters()))

    crf_transitions_params = []
    crf_ratio_params = []
    bert_params = []
    for params in crf_transitions_list:
        crf_transitions_params.append(params[1])
    for params in crf_ratio_list:
        crf_ratio_params.append(params[1])
    for params in bert_list:
        bert_params.append(params[1])

    optim = AdamW([{'params': crf_transitions_params, 'lr': args.crf_transition_lr},
                   {'params': crf_ratio_params, 'lr': args.crf_ratio_lr},
                   {'params': bert_params}], lr=args.bert_lr)
    total_steps = int(
        len(train_dataset) * args.epoch / (args.per_gpu_batch_size * args.gradient_accumulation_steps * args.n_gpu))
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(total_steps * 0.05),
                                                num_training_steps=total_steps)

    # training
    logger.info('Start Training')
    logger.info(args)
    logger.info('Total training batch size: {}'.format(args.per_gpu_batch_size * args.gradient_accumulation_steps * args.n_gpu))
    logger.info('Total Optimization Step: ' + str(total_steps))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_batch_size * args.n_gpu, sampler=train_sampler, num_workers=4, worker_init_fn=worker_init_fn)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    model.zero_grad()

    epochs_trained = 0
    train_iterator = trange(epochs_trained, args.epoch, desc="Epoch")

    set_seed(args.seed)  # add here for reproducibility

    best_metric = 0
    test_metric_with_best_valid = 0
    best_loss = 10000
    early_stop_count = 0

    global_steps = 0
    eval_steps = max(args.eval_steps, len(train_dataloader))

    loss_func = nn.CrossEntropyLoss()

    for epo, _ in enumerate(train_iterator):
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            optim.zero_grad()

            if args.TASK in ['seq', 'oos', 'da']:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['seq_labels'].to(args.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                cur_loss = outputs[0].mean()
            elif args.TASK == 'nli':
                s1_input_ids = batch['context']['input_ids'].to(args.device)
                s1_attention_mask = batch['context']['attention_mask'].to(args.device)
                s2_input_ids = batch['response']['input_ids'].to(args.device)
                s2_attention_mask = batch['response']['attention_mask'].to(args.device)
                labels = batch['seq_labels'].to(args.device)
                outputs = model(s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask, labels=labels)
                cur_loss = outputs[0]
            elif args.TASK == 'rs':
                context_input_ids = batch['context']['input_ids'].to(args.device)
                context_attention_mask = batch['context']['attention_mask'].to(args.device)
                response_input_ids = batch['response']['input_ids'].to(args.device)
                response_attention_mask = batch['response']['attention_mask'].to(args.device)
                context_output, response_output = model(context_input_ids, context_attention_mask, response_input_ids, response_attention_mask)
                logits = torch.mm(context_output, response_output.t().contiguous())
                batch_size = logits.shape[0]
                labels = torch.tensor(np.arange(batch_size)).to(logits.device)
                cur_loss = loss_func(logits, labels)
            else:
                raise ValueError("Please choose task from ['seq', 'oos', 'nli', 'da', 'rs']")


            # evaluation
            if global_steps % eval_steps == 0:
                val_metric, val_loss = evaluate(val_dataset, model, args, prefix="Valid")
                val_metric = val_metric['main']
                if args.early_stop_type == "metric":
                    if val_metric <= best_metric:
                        early_stop_count += 1
                    else:
                        early_stop_count = 0
                        best_metric = val_metric
                        test_metric_with_best_valid, _ = evaluate(test_dataset, model, args, prefix="Test")
                        model_to_save = deepcopy(model.module) if hasattr(model, "module") else deepcopy(model)

                elif args.early_stop_type == "loss":
                    if val_loss >= best_loss:
                        early_stop_count += 1
                    else:
                        early_stop_count = 0
                        best_loss = val_loss
                        test_metric_with_best_valid, _ = evaluate(test_dataset, model, args, prefix="Test")
                        model_to_save = deepcopy(model.module) if hasattr(model, "module") else deepcopy(model)
                else:
                    raise ValueError("Choose early_stop_type from ['loss', 'metric']")


                logger.info("Early Stop Count: {} | {}".format(early_stop_count, args.patience))

            # update parameters
            global_steps += 1
            loss = cur_loss.mean()
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
            
            epoch_loss += loss.item()
            epoch_iterator.set_description("Training Loss: {:.4f}".format(epoch_loss/(step+1)))



        if early_stop_count > args.patience:
            logger.info("Early Stopped!")
            break


    with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
        f.write("\n")
    with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
        if args.TASK == 'oos':
            f.write('Accuracy: %.4f, In Acc: %.4f, Out Acc: %.4f, Out Recall: %.4f \n' % (test_metric_with_best_valid['main'], 
                                                                                        test_metric_with_best_valid['in_acc'], 
                                                                                        test_metric_with_best_valid['oos_acc'], 
                                                                                        test_metric_with_best_valid['oos_recall']))
        elif args.TASK == 'da':
            f.write('Micro F1: %.4f, Macro F1: %.4f \n' % (test_metric_with_best_valid['micro_f1'], 
                                                            test_metric_with_best_valid['macro_f1']))  
        elif args.TASK == 'rs':
            f.write('Recall@ 1: %.4f, 3: %.4f, 5: %.4f, 10: %.4f \n' % (test_metric_with_best_valid['recall_1'], test_metric_with_best_valid['recall_3'], test_metric_with_best_valid['recall_5'], test_metric_with_best_valid['recall_10']))
        else:
            f.write("%5f \n" % (test_metric_with_best_valid['main']))

    
    return test_metric_with_best_valid, model_to_save


    



def main():
    # config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default='data/',
        type=str,
        help="The input data dir. Should contain the event detection data",
    )
    parser.add_argument(
        "--model_type",
        default='bert-base-cased',
        type=str,
        help="Model type",
    )
    parser.add_argument(
        "--CRF", action="store_true", help="Whether use CRF or not"
    )
    parser.add_argument(
        "--TASK",
        default=None,
        type=str,
        required=True,
        help="choose from ['seq', 'rs', 'da'], 'seq' stands for sequence classification, stands for token classification"
             "'da' stands for dialogue action prediction",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length", default=256, type=int, help="Max sequence length for prediction"
    )
    parser.add_argument(
        "--bert_lr", default=5e-5, type=float, help="The peak learning rate for BERT."
    )
    parser.add_argument(
        "--crf_transition_lr", default=1e-4, type=float, help="The peak learning rate for CRF transition matrix."
    )
    parser.add_argument(
        "--crf_ratio_lr", default=1e-4, type=float, help="The peak learning rate for CRF ratio."
    )
    parser.add_argument(
        "--epoch", default=5, type=int, help="Number of epoch for training"
    )
    parser.add_argument(
        "--num_labels", default=12, type=int, help="Number of unique labels in the dataset"
    )
    parser.add_argument(
        "--per_gpu_batch_size", default=2, type=int, help="Batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=2, type=int, help="Batch size"
    )
    parser.add_argument(
        "--seed", default=24, type=int, help="Random seed"
    )
    parser.add_argument(
        "--n_gpu", default=4, type=int, help="Number of GPUs"
    )
    parser.add_argument(
        "--device", default='cpu', type=str, help="Number of GPUs"
    )

    # for few-shot
    parser.add_argument(
        "--data_ratio", default=1, type=float, help="Number of shot to use. -1 for full-shot."
    )
    parser.add_argument(
        "--num_runs", default=10, type=int, help="Number of runs for few-shot model training. (with different seeds for few-shot data sampling)"
    )
    parser.add_argument(
        "--eval_steps", default=10, type=int, help="Number of steps for evaluation"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Whether save the final model and tokenizer"
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="Patience for early stop"
    )
    parser.add_argument(
        "--early_stop_type", 
        type=str, 
        default="metric", 
        help="choose from ['loss', 'metric'] "
    )
    parser.add_argument(
        "--classification_pooling",
        default='average',
        type=str,
        help="choose from ['cls', 'average', 'cls_nopool']",
    )

    # for multiturn dialogue tasks (e.g., dialogues action predication and response selection)
    parser.add_argument(
        "--classifier_dropout", default=0.1, type=float, help="The dropout of the classification layer."
    )
    parser.add_argument(
        "--dialogue_pooling_method",
        default='rnn',
        type=str,
        help="choose from ['rnn', 'attention', 'multihead_attention', 'mean', 'max']",
    )
    parser.add_argument(
        "--num_turn", default=3, type=int, help="Number of past turns used in dialogues action predication and response selection"
    )
    parser.add_argument(
        "--concatenate", action="store_true", default=False, help="Whether to use the concatenation of multiturn dialogue as model input. \
                                                                    If set as False, will split dialogue history into each turn, represent each of \
                                                                    them and use dialogue_pooling_method to get a multi-turn dialogue representation"
    )




    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    # initialize
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    

    # load data
    logger.info('Processing and loading data')


    def load_model(args, num_seq_label, weights=None):
        def get_model_class(args):
            if args.TASK in ['seq', 'oos']:
                if 'distil' in args.model_type:
                    return DistilBertForSequenceClassification
                else:
                    return BertForSequenceClassification
            elif args.TASK == 'nli':
                if 'distil' in args.model_type:
                    return DistilBertForNaturalLanguageInference
                else:
                    return BertForNaturalLanguageInference
            elif args.TASK == 'da':
                if 'distil' in args.model_type:
                    return DistilBertMultiTurnForDialogueActionPredictionConcat
                else:
                    return BertMultiTurnForDialogueActionPredictionConcat
            elif args.TASK == 'rs':
                if 'distil' in args.model_type:
                    return DistilBertMultiTurnForResponseSelectionConcat
                else:
                    return BertMultiTurnForResponseSelectionConcat
            else:
                raise ValueError("Please choose TASK from ['seq', 'oos', 'nli', 'da', 'rs']")

        
        MODEL_CLASS = get_model_class(args)
        config = BertConfig.from_pretrained(args.model_type)
        config.num_labels = num_seq_label
        config.classifier_dropout = args.classifier_dropout
        args.num_labels = config.num_labels
        if args.TASK in ["da", "rs"]:
            model = MODEL_CLASS.from_pretrained(args.model_type, config=config, num_turn=args.num_turn, dialogue_pooling_method=args.dialogue_pooling_method)
        else:
            model = MODEL_CLASS.from_pretrained(args.model_type, config=config, weights=weights, pooling=args.classification_pooling)
        return model

    # set data_ratio as -1 for full-data training
    if args.data_ratio == -1:
        if args.TASK == "da":
            train_dataset, test_dataset, val_dataset, num_seq_label = get_dialogue_action_dataset(BERT_MODEL=args.model_type, file_path=args.data_dir, 
                                                                                                max_seq_length=args.max_seq_length, concatenate=args.concatenate, 
                                                                                                num_turn=args.num_turn)
            model = load_model(args, num_seq_label)
        elif args.TASK == "rs":
            train_dataset, test_dataset, val_dataset = get_response_selection_dataset(BERT_MODEL=args.model_type, file_path=args.data_dir, 
                                                                                                max_seq_length=args.max_seq_length, concatenate=args.concatenate, 
                                                                                                num_turn=args.num_turn)
            model = load_model(args, 0)
        elif args.TASK == "nli":
            train_dataset, test_dataset, val_dataset, num_seq_label, weights = get_nli_dataset(BERT_MODEL=args.model_type, file_path=args.data_dir, max_seq_length=args.max_seq_length)
            model = load_model(args, num_seq_label, weights=weights)
        else:
            train_dataset, test_dataset, val_dataset, num_seq_label, weights = get_intent_slot_dataset(BERT_MODEL=args.model_type, 
                                                                                                                    max_seq_length=args.max_seq_length, data_path=args.data_dir, 
                                                                                                                    TASK=args.TASK)
            weights = weights.to(args.device)
            model = load_model(args, num_seq_label, weights)

        test_metric_with_best_valid = train(args, model, train_dataset, val_dataset, test_dataset)

    else:
        # initialize all the possible metrics 
        test_metrics = 0
        test_in_acc = 0
        test_oos_acc = 0
        test_oos_recall = 0
        test_micro_f1 = 0
        test_macro_f1 = 0
        test_recall_1 = 0
        test_recall_3 = 0
        test_recall_5 = 0
        test_recall_10 = 0

        # sequentially perform experiments on each few-shot dataset and record the results
        for idx in range(args.num_runs):
            DATA_DIR = os.path.join(args.data_dir, str(int(args.data_ratio)))
            DATA_DIR = os.path.join(DATA_DIR, str(idx))
            if args.TASK == "da":
                train_dataset, test_dataset, val_dataset, num_seq_label = get_dialogue_action_dataset(BERT_MODEL=args.model_type, file_path=DATA_DIR, 
                                                                                                max_seq_length=args.max_seq_length, concatenate=args.concatenate, 
                                                                                                num_turn=args.num_turn)
                model = load_model(args, num_seq_label)
            elif args.TASK == "rs":
                train_dataset, test_dataset, val_dataset = get_response_selection_dataset(BERT_MODEL=args.model_type, file_path=DATA_DIR, 
                                                                                                    max_seq_length=args.max_seq_length, concatenate=args.concatenate, 
                                                                                                    num_turn=args.num_turn)
                model = load_model(args, 0)
            elif args.TASK == "nli":
                train_dataset, test_dataset, val_dataset, num_seq_label, weights = get_nli_dataset(BERT_MODEL=args.model_type, file_path=DATA_DIR, max_seq_length=args.max_seq_length)
                model = load_model(args, num_seq_label, weights=weights)
            elif args.TASK in ['seq', 'oos']:
                train_dataset, test_dataset, val_dataset, num_seq_label, weights = get_intent_slot_dataset(BERT_MODEL=args.model_type, max_seq_length=args.max_seq_length, data_path=DATA_DIR, TASK=args.TASK)
                weights = weights.to(args.device)
                model = load_model(args, num_seq_label, weights)
            else:
                raise ValueError("Please choose task from ['seq', 'oos', 'nli', 'da', 'rs']")
            
            test_metric_with_best_valid, model_to_save = train(args, model, train_dataset, val_dataset, test_dataset)

            # save model
            if args.save_model:
                logger.info("Saving model checkpoint to %s", args.output_dir)
                model_to_save.save_pretrained(args.output_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.model_type)
                tokenizer.save_pretrained(args.output_dir)
            test_metrics += test_metric_with_best_valid['main']

            if args.TASK == "oos":
                test_in_acc += test_metric_with_best_valid['in_acc']
                test_oos_acc += test_metric_with_best_valid['oos_acc']
                test_oos_recall += test_metric_with_best_valid['oos_recall']
            elif args.TASK == "da":
                test_micro_f1 += test_metric_with_best_valid['micro_f1']
                test_macro_f1 += test_metric_with_best_valid['macro_f1']
            elif args.TASK == "rs":
                test_recall_1 += test_metric_with_best_valid['recall_1']
                test_recall_3 += test_metric_with_best_valid['recall_3']
                test_recall_5 += test_metric_with_best_valid['recall_5']
                test_recall_10 += test_metric_with_best_valid['recall_10']

        if args.TASK == "oos":
            test_metrics /= args.num_runs
            test_in_acc /= args.num_runs
            test_oos_acc /= args.num_runs
            test_oos_recall /= args.num_runs
            with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
                f.write("Average acc: %5f \n" % (test_metrics))
                f.write("Average in acc: %5f \n" % (test_in_acc))
                f.write("Average out acc: %5f \n" % (test_oos_acc))
                f.write("Average out recall: %5f \n" % (test_oos_recall))
        elif args.TASK == "da":
            test_metrics /= args.num_runs
            test_macro_f1 /= args.num_runs
            test_micro_f1 /= args.num_runs
            with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
                f.write("Average Micro F1: %5f \n" % (test_micro_f1))
                f.write("Average Macro F1: %5f \n" % (test_macro_f1))
        elif args.TASK == "rs":
            test_metrics /= args.num_runs
            test_recall_1 /= args.num_runs
            test_recall_3 /= args.num_runs
            test_recall_5 /= args.num_runs
            test_recall_10 /= args.num_runs
            with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
                f.write("Average Recall@1: %5f \n" % (test_recall_1))
                f.write("Average Recall@3: %5f \n" % (test_recall_3))
                f.write("Average Recall@5: %5f \n" % (test_recall_5))
                f.write("Average Recall@10: %5f \n" % (test_recall_10))
        else:
            test_metrics /= args.num_runs
            with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
                f.write("Average: %5f \n" % (test_metrics))


    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()