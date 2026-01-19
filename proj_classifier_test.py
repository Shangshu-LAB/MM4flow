from zat.log_to_dataframe import LogToDataFrame
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_model
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
import random
import json

class Classifier(nn.Module):
    def __init__(self, ps_config, bytes_config, num_classes):
        super(Classifier, self).__init__()
        self.ps_config, self.bytes_config = ps_config, bytes_config
        self.ps_encoder = BertForMaskedLM(ps_config)
        self.bytes_encoder = BertForMaskedLM(bytes_config)
        self.ps_cross_attention = nn.MultiheadAttention(embed_dim=ps_config.hidden_size, num_heads=4, batch_first=True)
        self.bytes_cross_attention = nn.MultiheadAttention(embed_dim=bytes_config.hidden_size, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(ps_config.hidden_size + bytes_config.hidden_size, num_classes)
        )
    def weight_init(self, premodel_ps_path, premodel_bytes_path):
        # self.base_model.load_state_dict(torch.load(os.path.join(premodel_path,f"checkpoint-{checkpoint}","pytorch_model.bin")))
        load_model(self.ps_encoder, os.path.join(premodel_ps_path, 'model.safetensors'), strict=False)
        load_model(self.bytes_encoder, os.path.join(premodel_bytes_path, 'model.safetensors'), strict=False)
    def forward(self, inputs):
        r = self.forward_with_attention_weight(inputs)
        return {'y_logit': r['y_logit']}
    def forward_with_attention_weight(self, inputs):
        ps_outputs = self.ps_encoder.bert(input_ids=inputs['ps'], attention_mask=inputs['ps_attention_mask'])
        raw_outputs = self.bytes_encoder.bert(input_ids=inputs['raw'], attention_mask=inputs['raw_attention_mask'], token_type_ids=inputs['raw_token_type_ids'])

        outputs = torch.concat([ps_outputs.last_hidden_state, raw_outputs.last_hidden_state], dim=1)
        key_padding_mask = (1 - torch.concat([inputs['ps_attention_mask'], inputs['raw_attention_mask']], dim=1)).bool()
        ps_attn_output, ps_attn_output_weights = self.ps_cross_attention(ps_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask)
        raw_attn_output, raw_attn_output_weights = self.bytes_cross_attention(raw_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask)
        
        memory_ps, memory_raw = ps_attn_output[:, 0, :], raw_attn_output[:, 0, :]
        y_logit = self.classifier(torch.concat([memory_ps, memory_raw], dim=1))
        return {'y_logit':y_logit, 'ps_attn_weights': ps_attn_output_weights, 'raw_attn_weights': raw_attn_output_weights}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(inputs)
        loss_cls = F.cross_entropy(output['y_logit'], inputs['y_label'])
        loss = loss_cls
        return (loss, {'y_logit':output['y_logit']}) if return_outputs else loss

def compute_metrics(pred):
    y_label = pred.label_ids[-1]
    y_pred = pred.predictions
    acc_y = (y_label==y_pred).mean()
    return {'accuracy_y':acc_y}
def preprocess_logits_for_metrics(logits, labels):
    y_logit = logits
    y_pred = y_logit.argmax(dim=-1)
    return y_pred  #, labels


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPUs is available.")
else:
    gpu_count = 1


import argparse


info_features = ['ts','id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service']
conn_features = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'conn_state']
ps_features = ['up', 'down', 'ps']
raw_features = ['fwd_raw', 'bwd_raw']

log2df = LogToDataFrame()

columns = ['fwd_raw', 'bwd_raw', 'ps']


max_length_ps= 256
max_length_bytes= 512
def func_ps(ps):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_length_ps:
            break
    return ' '.join(r[0:max_length_ps])
def func_bytes(s):
    return ' '.join([s[i:i+2] for i in range(0, len(s), 2)])
drop_empty = lambda x: x if x!='(empty)' else np.nan

tokenizer_ps = BertTokenizerFast.from_pretrained('tokenizer_bert/ps_tokenizer')
tokenizer_bytes = BertTokenizerFast.from_pretrained('tokenizer_bert/bytes_tokenizer')

premodel_name_ps = 'BERT-ps'
premodel_name_raw = 'BERT-bytes'

model_name = "MM4flow"

min_pkts = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='dataset')
    parser.add_argument('--model_ts', '-m', type=str, default='202412110000')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--output', '-o', type=str, default='result.csv')

    args = parser.parse_args()
    rootdir = args.dataset
    model_ts = args.model_ts
    batch_size = args.batch_size
    output = args.output

    def encode(examples):
        ps = tokenizer_ps(examples['ps'], truncation=True, padding="max_length", max_length=max_length_ps, return_special_tokens_mask=True)
        raw = tokenizer_bytes(list(zip(examples['fwd_raw'], examples['bwd_raw'])), truncation=True, padding="max_length", max_length=max_length_bytes, return_special_tokens_mask=True)
        return {
            'ps': ps['input_ids'], 'ps_attention_mask': ps['attention_mask'],
            'raw': raw['input_ids'], 'raw_attention_mask': raw['attention_mask'], 'raw_token_type_ids': raw['token_type_ids'],
        }

    df_test = pd.DataFrame()
    for filename in tqdm(os.listdir(rootdir), desc='reading pcap_log'):
        # print('\t', filename, end='\t')
        log_path = os.path.join(rootdir, filename)
        conn = log2df.create_dataframe(os.path.join(log_path, 'conn.log'), ts_index=False).set_index('uid')
        ps = log2df.create_dataframe(os.path.join(log_path, 'ps.log'), ts_index=False).set_index('uid')
        raw = log2df.create_dataframe(os.path.join(log_path, 'raw.log'), ts_index=False).set_index('uid')
        index = list(set(conn.index) & set(ps.index) & set(raw.index))
        df_tmp = pd.concat(
            [conn.loc[index][info_features + conn_features], ps.loc[index][ps_features], raw.loc[index][raw_features]],
            axis=1)
        df_test = pd.concat([df_test, df_tmp])

    df_test = df_test[df_test['up'] + df_test['down'] >= min_pkts]
    df_test['ps'] = df_test['ps'].apply(func_ps)
    df_test['fwd_raw'] = df_test['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_test['bwd_raw'] = df_test['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    testset = Dataset.from_pandas(df_test[columns])
    testset = testset.map(encode, batched=True)
    testset = testset.remove_columns(['fwd_raw', 'bwd_raw', 'uid'])
    testset.set_format(type='torch', columns=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids'])

    model_path = os.path.join('model-classifier', model_ts)

    with open(os.path.join(model_path, model_name, "info.json")) as f:
        info = json.load(f)
    label2idx = info["label2idx"]
    num_classes = len(label2idx)
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))

    pre_timestamp_ps, pre_timestamp_raw = info["pre_timestamp_ps"], info["pre_timestamp_raw"]
    with open(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}', 'hyperparameters.json')) as f:
        hyperparameters_ps = json.load(f)
    with open(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}', 'hyperparameters.json')) as f:
        hyperparameters_bytes = json.load(f)

    ps_config = BertConfig(
        vocab_size=len(tokenizer_ps.get_vocab()), max_position_embeddings=max_length_ps,
        hidden_size=hyperparameters_ps['d_model'], num_hidden_layers=hyperparameters_ps['n_layer'],
        num_attention_heads=hyperparameters_ps['n_head'], intermediate_size=hyperparameters_ps['dim_ff'],
    )
    bytes_config = BertConfig(
        vocab_size=len(tokenizer_bytes.get_vocab()), max_position_embeddings=max_length_bytes,
        hidden_size=hyperparameters_bytes['d_model'], num_hidden_layers=hyperparameters_bytes['n_layer'],
        num_attention_heads=hyperparameters_bytes['n_head'], intermediate_size=hyperparameters_bytes['dim_ff'],
    )

    training_results = []
    model_type = 'finetune'

    model_dir = os.path.join(model_path, model_name, model_type)
    model = Classifier(ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
    model.eval()

    y_pred = []
    for i in tqdm(range(0, testset.num_rows, batch_size), desc='Detection'):
        batch_inputs = {k: v.to(device) for k, v in testset[i:i + batch_size].items()}
        y_pred_tmp = model(batch_inputs)['y_logit'].argmax(1).cpu().numpy()
        y_pred = np.concatenate([y_pred, y_pred_tmp])

    df_test['pred'] = y_pred
    df_test['pred'] = df_test['pred'].map(idx2label)
    print(df_test['pred'].value_counts())

    df_test[info_features+conn_features+['pred']].to_csv(output)

    print(f"The analysis results are saved in {output}.")
