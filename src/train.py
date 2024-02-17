import os
import random
import datetime
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding
import model
from model import Causal_Model
from utils import setup_seed, compute_metrics, record_best_scores, negative_sampling

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dataset/ESC/ESC_dataset')
parser.add_argument('--dataset_name', type=str, default='ESC', help='name used to save checkpoints')
parser.add_argument('--num_folds', type=int, default=5, help='conduct n-fold cross validation')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--train_batchsize', type=int, default=20)
parser.add_argument('--test_batchsize', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--bert_path', type=str, default='bert-large-uncased')
parser.add_argument('--d_model', type=int, default=1024, help='hidden dimension of the model')
parser.add_argument('--num_heads', type=int, default=16, help='number of heads in multi-head attention')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='drop out rate of FFN in the model')
parser.add_argument('--visualize', type=bool, default=False, help='demonstrate the generated token and sentence')
parser.add_argument('--SEED', type=int, default=3407)
parser.add_argument('--shuffle', type=bool, default=False, help='if shuffle==False, use cross-topic partition (ESC).\
                    If shuffle==True, random partition (ESC*)')

args, _ = parser.parse_known_args()
dataset=args.dataset
dataset_name=args.dataset_name
num_folds=args.num_folds
num_epochs=args.num_epochs
learning_rate=args.learning_rate
train_btz=args.train_batchsize
test_btz=args.test_batchsize
bert_path=args.bert_path
d_model=args.d_model
num_heads=args.num_heads
dropout_rate=args.dropout_rate
visualize=args.visualize
SEED=args.SEED
shuffle=args.shuffle

setup_seed(SEED)


checkpoint = bert_path
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
special_tokens_dict = {'additional_special_tokens': ['<e1>','</e1>','<e2>','</e2>']}
tokenizer.add_special_tokens(special_tokens_dict)

def tokenize_function_mask(examples):
    return tokenizer(examples["event_masked_sentence"], truncation=True)

def tokenize_function_tag(examples):
    return tokenizer(examples["event_tagged_sentence"], truncation=True)


total_dataset = load_from_disk(dataset)
device = 'cuda'
fold_size = len(total_dataset) // num_folds

checkpoint_path = f'checkpoints/{dataset_name}_{bert_path}'
os.makedirs(checkpoint_path, exist_ok=True)
print(f"Save the checkpoint at: {checkpoint_path}")

if dataset_name == 'ESC' and shuffle:
    total_dataset = total_dataset.shuffle(seed=SEED)
    print('\n*************************************************************')
    print('*********     use random partition (ESC*)    ****************')
    print('*************************************************************\n')
    
else:
    print('\n*************************************************************')
    print('*********     use sorted partition (ESC)    ****************')
    print('*************************************************************\n')


for i in range(num_folds):
    print(f"start Fold {i+1} training")
    
    wandb.init(project="HSemCD", name=f'{dataset_name}_{i+1}')
   
    # dataset
    test_indices = list(range(i * fold_size, (i + 1) * fold_size))
    train_indices = list(set(range(len(total_dataset))) - set(test_indices))
    train_fold = total_dataset.select(train_indices)
    if dataset_name=='CTB':
        train_fold = train_fold.shuffle(seed=SEED)
        train_fold = train_fold.filter(negative_sampling) 
    
    # train
    masked_train_fold = train_fold.map(tokenize_function_mask, batched=True, batch_size=32)
    masked_train_fold = masked_train_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
    masked_train_fold.set_format("torch")
    
    tagged_train_fold = train_fold.map(tokenize_function_tag, batched=True, batch_size=32)
    tagged_train_fold = tagged_train_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
    tagged_train_fold.set_format("torch")
    
    # test
    test_fold = total_dataset.select(test_indices)
    masked_test_fold = test_fold.map(tokenize_function_mask, batched=True, batch_size=32)
    masked_test_fold = masked_test_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
    masked_test_fold.set_format("torch")
    
    tagged_test_fold = test_fold.map(tokenize_function_tag, batched=True, batch_size=32)
    tagged_test_fold = tagged_test_fold.remove_columns(['sentence', 'event_tagged_sentence', 'event_masked_sentence','e1','e2'])
    tagged_test_fold.set_format("torch")
    
    print(f"train len: {len(masked_train_fold)}, test len: {len(masked_test_fold)}")
    
    #dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader_mask_train = DataLoader(
    masked_train_fold, shuffle=False, batch_size=train_btz, collate_fn=data_collator)
    
    dataloader_tag_train = DataLoader(
    tagged_train_fold, shuffle=False,  batch_size=train_btz, collate_fn=data_collator)
    
    dataloader_mask_test = DataLoader(
    masked_test_fold, shuffle=False, batch_size=test_btz, collate_fn=data_collator)
    
    dataloader_tag_test = DataLoader(
    tagged_test_fold, shuffle=False,  batch_size=test_btz, collate_fn=data_collator)

    dataloader_mask_train = tqdm(dataloader_mask_train, dynamic_ncols=True)
    dataloader_mask_test = tqdm(dataloader_mask_test, dynamic_ncols=True)

    #===============
    model=Causal_Model(bert_path, d_model, num_heads, dropout_rate, device, visualize)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion=nn.CrossEntropyLoss()
    model=model.to(device)

    # F1_flag
    highest_f1=0.0
    
    for epoch in range(num_epochs):
        #==============
        #==== train ===
        #==============
        model.train()
        predicted_all=[]
        gold_all=[]
        mean_loss = torch.zeros(1).to(device)
        iteration=0

        for mask_data, tag_data in zip(dataloader_mask_train, dataloader_tag_train):
           
            mask_data, tag_data=mask_data.to(device), tag_data.to(device)
            labels = tag_data['labels']
            labels = labels.to(device)
            del mask_data['labels']
            del tag_data['labels']
            
            outputs=model(mask_data, tag_data).squeeze(1)
            loss = criterion(outputs, labels)
            mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
            iteration+=1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            predicted = torch.argmax(outputs, dim=-1)
            predicted = list(predicted.cpu().numpy())
            predicted_all += predicted
            gold_all += list(labels.cpu().numpy())
                                                                      
        precision, recall, f1_score = compute_metrics(gold_all, predicted_all)
        print(f"[epoch {epoch+1}| train] p:{precision*100:.2f} r:{recall*100:.2f} F1:{f1_score*100:.2f} loss:{mean_loss.item():.4f}")
        
        
        #==============
        #==== test ====
        #==============
        model.eval()
        mean_loss_test = 0
        predicted_all_test = []
        gold_all_test = []
        with torch.no_grad():
            iteration=0
            for mask_data, tag_data in zip(dataloader_mask_test, dataloader_tag_test):
                
                mask_data, tag_data=mask_data.to(device), tag_data.to(device)
                labels=tag_data['labels']
                labels=labels.to(device)
                del mask_data['labels']
                del tag_data['labels']
                
                outputs=model(mask_data, tag_data).squeeze(1)
                loss = criterion(outputs, labels)
                mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)
                iteration+=1

                predicted = torch.argmax(outputs, dim=-1)
                predicted=list(predicted.cpu().numpy())
                predicted_all_test+=predicted
                gold_all_test+=list(labels.cpu().numpy())
                                                          
        precision_t, recall_t, f1_score_t = compute_metrics(gold_all_test, predicted_all_test)
        print(f"[epoch {epoch+1}| test] p:{precision_t*100:.2f} r:{recall_t*100:.2f} F1:{f1_score_t*100:.2f} loss:{mean_loss_test.item():.4f}")
        
        # log to wandb
        wandb.log({
            "Epoch": epoch,
            "test_F1": f1_score_t*100,
            "test_recall": recall_t*100,
            "test_precision": precision_t*100,
            "test_loss": mean_loss_test.item(),
            "train_F1": f1_score*100,
            "train_recall": recall*100,
            "train_precision": precision*100,
            "train_loss": mean_loss.item(),
        })
        
        # save the model
        if f1_score_t*100 > highest_f1:
            highest_f1= f1_score_t*100
            torch.save(model.state_dict(), checkpoint_path + f'/best_model_fold{i+1}.pt')
            print(f"Current highest F1: {highest_f1:.2f}, checkpoint saved.")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            record_best_scores(current_time, precision_t, recall_t, f1_score_t, checkpoint_path + f'/best_scores_fold{i+1}.txt')
        
    print(f"End Fold {i+1} training")
    wandb.finish()








