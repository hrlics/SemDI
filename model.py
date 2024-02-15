from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import random


class ClozeAnalyzer(nn.Module):
    def __init__(self, tokenizer, bert, device, visualize = False):
        super(ClozeAnalyzer, self).__init__()
        self.tokenizer = tokenizer
        self.bert = bert
        self.visualize = visualize
        self.device = device
        
    def forward(self, x, groundtruth):

        token_logits = self.bert(**x).logits # z: (batch_size, length, vocab_size)
        mask_token_index = torch.where(x['input_ids'] == self.tokenizer.mask_token_id)[1] # z: (btz)
        mask_token_logits = torch.stack( 
            [ token_logits[idx, mask_token_index[idx], :] for idx in range(x['input_ids'].size(0))], 
            dim=0) # z: (batch_size, vocab_size)
        
        
        gen_token = [ self.tokenizer.decode(token_id) for token_id in torch.argmax(mask_token_logits, dim=-1) ]
        og_mask_sentences = [ self.tokenizer.decode(sequence[1:-1]) for sequence in x['input_ids'] ] 
        gen_sentences = [sentence.replace(self.tokenizer.mask_token, gen_token[idx]) for idx, sentence in enumerate( og_mask_sentences)]
        if self.visualize:
            og_sentences=[ self.tokenizer.decode(sequence[1:-1]) for sequence in groundtruth['input_ids'] ] 
            print(f"generated tokens: {gen_token}")
            print(f"original sentences: {og_sentences}")
            print(f"original mask sentences: {og_mask_sentences}")
            print(f"generated sentences: {gen_sentences}")

        outputs = self.bert.base_model(
            **self.tokenizer(gen_sentences, return_tensors="pt", padding=True).to(self.device)
            ).last_hidden_state # z: (batch_size, seq_len, d_model)
        
        ret = torch.stack( 
            [outputs[idx, mask_token_index[idx], :] for idx in range(x['input_ids'].size(0))], 
            dim=0).unsqueeze(1) # z: (batch_size, 1, d_model)
        
        return ret 
        

class Discriminator(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, tokenizer, bert, device):
        super(Discriminator, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.tokenizer =  tokenizer
        self.bert = bert
        self.device= device
        # FFN
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.fc3 = nn.Linear(d_model, 2)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, groundtruth):
        
        key = self.bert.base_model(**groundtruth).last_hidden_state.permute(1, 0, 2)
        value =key
        x = x.permute(1, 0, 2)
        attn_output, attn_weights = self.mha(x, key, value)
        attn_output = attn_output.permute(1, 0, 2) # z: (batch_size, 1, embedding)

        # FFN
        out=self.dropout(self.relu(self.fc1(attn_output)))
        out=self.fc2(out)
        out=self.layer_norm(attn_output+out)
        out=self.fc3(out) 
        
        return out
      

    

class Causal_Model(nn.Module):
    def __init__(self, bert_path, d_model, num_heads, dropout_rate, device, visualize=False):
        super(Causal_Model, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        special_tokens_dict = {'additional_special_tokens': ['<e1>','</e1>','<e2>','</e2>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        
        self.generator = ClozeAnalyzer(self.tokenizer, self.bert, device, visualize)
        self.discriminator = Discriminator(d_model, num_heads, dropout_rate, self.tokenizer, self.bert, device)
        

    def forward(self, x, groundtruth):
        
        out=self.generator(x, groundtruth) 
        out=self.discriminator(out, groundtruth)
        return out