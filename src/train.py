import os
import random
import wandb
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding

import model
from model import Causal_Model
from utils import setup_seed, compute_metrics, record_best_scores, negative_sampling


def train(args, model, optimizer, criterion, dataloader_mask_train, dataloader_tag_train, device):
    model.train()
    predicted_all = []
    gold_all = []
    mean_loss = torch.zeros(1).to(device)
    iteration = 0

    for mask_data, tag_data in zip(dataloader_mask_train, dataloader_tag_train):
        mask_data, tag_data = mask_data.to(device), tag_data.to(device)
        labels = tag_data['labels'].to(device)
        del mask_data['labels']
        del tag_data['labels']
        
        outputs = model(mask_data, tag_data).squeeze(1)
        loss = criterion(outputs, labels)
        mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
        iteration += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        predicted = torch.argmax(outputs, dim=-1)
        predicted_all += list(predicted.cpu().numpy())
        gold_all += list(labels.cpu().numpy())
    
    precision, recall, f1_score = compute_metrics(gold_all, predicted_all)
    return precision, recall, f1_score, mean_loss.item()


def evaluate(args, model, criterion, dataloader_mask_test, dataloader_tag_test, device):
    model.eval()
    mean_loss_test = 0
    predicted_all_test = []
    gold_all_test = []
    with torch.no_grad():
        iteration = 0
        for mask_data, tag_data in zip(dataloader_mask_test, dataloader_tag_test):
            mask_data, tag_data = mask_data.to(device), tag_data.to(device)
            labels = tag_data['labels'].to(device)
            del mask_data['labels']
            del tag_data['labels']
            
            outputs = model(mask_data, tag_data).squeeze(1)
            loss = criterion(outputs, labels)
            mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)
            iteration += 1

            predicted = torch.argmax(outputs, dim=-1)
            predicted_all_test += list(predicted.cpu().numpy())
            gold_all_test += list(labels.cpu().numpy())
    
    precision, recall, f1_score = compute_metrics(gold_all_test, predicted_all_test)
    return precision, recall, f1_score, mean_loss_test.item()


def run_training(args):    
    dataset_name = args.dataset_name
    setup_seed(args.SEED)
    
    # add event specification symbols
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    special_tokens_dict = {'additional_special_tokens': ['<e1>','</e1>','<e2>','</e2>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # load dataset
    total_dataset = load_from_disk(args.dataset)
    print(f'\ndataset: {dataset_name}')
    if dataset_name == 'ESC_star' and args.shuffle:
        total_dataset = total_dataset.shuffle(seed=args.SEED)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\ndevice: {device}')

    fold_size = len(total_dataset) // args.num_folds
    checkpoint_path = f'checkpoints/{dataset_name}_{args.bert_path.split("/")[1]}'
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"\nSave the checkpoint at: {checkpoint_path}")

    
    for i in range(args.num_folds):
        print(f"\nstart Fold {i+1} training")
        wandb.init(project="SemDI", name=f'{args.dataset_name}_fold{i+1}')

        # dataset
        test_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(len(total_dataset))) - set(test_indices))
        train_fold = total_dataset.select(train_indices)
        if dataset_name == 'CTB':
            train_fold = train_fold.shuffle(seed=args.SEED)
            train_fold = train_fold.filter(negative_sampling)

        # train fold
        masked_train_fold = train_fold.map(lambda x: tokenizer(x["event_masked_sentence"], truncation=True), batched=True, batch_size=32)
        masked_train_fold = masked_train_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
        masked_train_fold.set_format("torch")
        
        tagged_train_fold = train_fold.map(lambda x: tokenizer(x["event_tagged_sentence"], truncation=True), batched=True, batch_size=32)
        tagged_train_fold = tagged_train_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
        tagged_train_fold.set_format("torch")

        # test fold
        test_fold = total_dataset.select(test_indices)
        masked_test_fold = test_fold.map(lambda x: tokenizer(x["event_masked_sentence"], truncation=True), batched=True, batch_size=32)
        masked_test_fold = masked_test_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
        masked_test_fold.set_format("torch")
        
        tagged_test_fold = test_fold.map(lambda x: tokenizer(x["event_tagged_sentence"], truncation=True), batched=True, batch_size=32)
        tagged_test_fold = tagged_test_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
        tagged_test_fold.set_format("torch")

        print(f"\ntrain length: {len(masked_train_fold)}, test length: {len(masked_test_fold)}")

        # dataloader
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        dataloader_mask_train = DataLoader(masked_train_fold, shuffle=False, batch_size=args.train_batchsize, collate_fn=data_collator)
        dataloader_tag_train = DataLoader(tagged_train_fold, shuffle=False, batch_size=args.train_batchsize, collate_fn=data_collator)
        dataloader_mask_test = DataLoader(masked_test_fold, shuffle=False, batch_size=args.test_batchsize, collate_fn=data_collator)
        dataloader_tag_test = DataLoader(tagged_test_fold, shuffle=False, batch_size=args.test_batchsize, collate_fn=data_collator)
        dataloader_mask_train = tqdm(dataloader_mask_train, dynamic_ncols=True)
        dataloader_mask_test = tqdm(dataloader_mask_test, dynamic_ncols=True)

        # model
        model = Causal_Model(args.bert_path, args.d_model, args.num_heads, args.dropout_rate, device, args.visualize)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)

        # run
        highest_f1 = 0.0
        for epoch in range(args.num_epochs):
            train_precision, train_recall, train_f1, train_loss = train(args, model, optimizer, criterion, dataloader_mask_train, dataloader_tag_train, device)
            test_precision, test_recall, test_f1, test_loss = evaluate(args, model, criterion, dataloader_mask_test, dataloader_tag_test, device)
            
            print(f"[epoch {epoch+1}| train] p:{train_precision*100:.2f} r:{train_recall*100:.2f} F1:{train_f1*100:.2f} loss:{train_loss:.4f}")
            print(f"[epoch {epoch+1}| test] p:{test_precision*100:.2f} r:{test_recall*100:.2f} F1:{test_f1*100:.2f} loss:{test_loss:.4f}")
            
            wandb.log({
                "Epoch": epoch,
                "test_F1": test_f1*100,
                "test_recall": test_recall*100,
                "test_precision": test_precision*100,
                "test_loss": test_loss,
                "train_F1": train_f1*100,
                "train_recall": train_recall*100,
                "train_precision": train_precision*100,
                "train_loss": train_loss,
            })
            
            if test_f1*100 > highest_f1:
                highest_f1 = test_f1*100
                torch.save(model.state_dict(), checkpoint_path + f'/best_model_fold{i+1}.pt')
                print(f"Current highest F1: {highest_f1:.2f}, checkpoint saved.")
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                record_best_scores(current_time, test_precision, test_recall, test_f1, checkpoint_path + f'/best_scores_fold{i+1}.txt')
        
        print(f"End Fold {i+1} training")
        wandb.finish()


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/ESC/ESC_dataset')
    parser.add_argument('--dataset_name', type=str, default='ESC', help='name used to save checkpoints')
    parser.add_argument('--num_folds', type=int, default=5, help='conduct n-fold cross validation')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batchsize', type=int, default=20)
    parser.add_argument('--test_batchsize', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--bert_path', type=str, default='FacebookAI/roberta-large')
    parser.add_argument('--d_model', type=int, default=1024, help='hidden dimension of the model')
    parser.add_argument('--num_heads', type=int, default=16, help='number of heads in multi-head attention')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='drop out rate of FFN in the model')
    parser.add_argument('--visualize', action='store_true', help='demonstrate the generated token')
    parser.add_argument('--SEED', type=int, default=3407)
    parser.add_argument('--shuffle', action='store_true', help='if shuffle=False, use cross-topic partition(ESC).\
                        If shuffle=True, random partition(ESC*)')

    args = parser.parse_args()
    run_training(args)
