from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_model
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
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

columns = ['fwd_raw', 'bwd_raw', 'ps', 'y_label']

min_pkts = 5
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
pre_timestamp_ps = '202406132201'
pre_timestamp_raw = '202407081419'

model_name = "MM4flow"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-T', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='dataset')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_epochs', '-ep', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5)

    args = parser.parse_args()
    timestamp = args.timestamp
    dataset_path = args.dataset
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    output = args.output

    model_root = os.path.join('model-classifier', output) if output is not None else 'model-classifier'
    if not os.path.exists(model_root):
        os.mkdir(model_root)

    df_train = pd.read_csv(f'{dataset_path}/train.csv.gz', compression='gzip', index_col=0)
    df_train['ps'] = df_train['ps'].apply(func_ps)
    df_train['fwd_raw'] = df_train['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_train['bwd_raw'] = df_train['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)

    labels = list(df_train['label'].value_counts().index)
    label2idx = dict(zip(labels, range(len(labels))))
    df_train['y_label'] = df_train['label'].map(label2idx)
    num_classes = len(label2idx)

    def encode(examples):
        ps = tokenizer_ps(examples['ps'], truncation=True, padding="max_length", max_length=max_length_ps, return_special_tokens_mask=True)
        raw = tokenizer_bytes(list(zip(examples['fwd_raw'], examples['bwd_raw'])), truncation=True, padding="max_length", max_length=max_length_bytes, return_special_tokens_mask=True)
        return {
            'ps': ps['input_ids'], 'ps_attention_mask': ps['attention_mask'],
            'raw': raw['input_ids'], 'raw_attention_mask': raw['attention_mask'], 'raw_token_type_ids': raw['token_type_ids'],
        }

    trainset = Dataset.from_pandas(df_train[columns])
    trainset = trainset.map(encode, batched=True)
    trainset = trainset.remove_columns(['fwd_raw', 'bwd_raw', 'uid'])
    trainset.set_format(type='torch', columns=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label'])
    trainset = trainset.shuffle()

    df_eval = pd.read_csv(f'{dataset_path}/val.csv.gz', compression='gzip', index_col=0)
    df_eval['ps'] = df_eval['ps'].apply(func_ps)
    df_eval['fwd_raw'] = df_eval['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_eval['bwd_raw'] = df_eval['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_eval['y_label'] = df_eval['label'].map(label2idx)
    evalset = Dataset.from_pandas(df_eval[columns])
    evalset = evalset.map(encode, batched=True)
    evalset = evalset.remove_columns(['fwd_raw', 'bwd_raw', 'uid'])
    evalset.set_format(type='torch', columns=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label'])
    evalset = evalset.shuffle()

    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', compression='gzip', index_col=0)
    df_test['ps'] = df_test['ps'].apply(func_ps)
    df_test['fwd_raw'] = df_test['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_test['bwd_raw'] = df_test['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    df_test['y_label'] = df_test['label'].map(label2idx)
    testset = Dataset.from_pandas(df_test[columns])
    testset = testset.map(encode, batched=True)
    testset = testset.remove_columns(['fwd_raw', 'bwd_raw', 'uid'])
    testset.set_format(type='torch', columns=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label'])
    testset = testset.shuffle()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if not os.path.exists(os.path.join(model_root, timestamp)):
        os.mkdir(os.path.join(model_root, timestamp))

    info = {
        'dataset': dataset_path,
        'premodel_name_ps': premodel_name_ps, 'pre_timestamp_ps': pre_timestamp_ps,
        'premodel_name_raw': premodel_name_raw, 'pre_timestamp_raw': pre_timestamp_raw,
        'num_classes': num_classes, 'label2idx': label2idx
    }
    with open(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}', 'hyperparameters.json')) as f:
        hyperparameters_ps = json.load(f)
    with open(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}', 'hyperparameters.json')) as f:
        hyperparameters_bytes = json.load(f)
    print("MM4flow-ps", hyperparameters_ps)
    print("MM4flow-raw", hyperparameters_bytes)

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
    premodel_ps = BertForMaskedLM(ps_config).to(device)
    premodel_raw = BertForMaskedLM(bytes_config).to(device)
    load_model(premodel_ps, os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}', 'model.safetensors'), strict=False)
    load_model(premodel_raw, os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}', 'model.safetensors'), strict=False)

    training_results = []
    if not os.path.exists(os.path.join(model_root, timestamp, model_name)):
    	os.mkdir(os.path.join(model_root, timestamp, model_name))
    with open(os.path.join(model_root, timestamp, model_name, 'info.json'), 'w') as f:
        json.dump(info, f)

    model_type = 'finetune'
    model_dir = os.path.join(model_root, timestamp, model_name, model_type)
    print(model_dir)
    model = Classifier(ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes).to(device)
    model.weight_init(
        premodel_ps_path=os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}'),
        premodel_bytes_path=os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}')
    )
    model.ps_encoder.requires_grad_(requires_grad=False)
    model.bytes_encoder.requires_grad_(requires_grad=False)
    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='no',
        label_names=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label'],
        bf16=True,
    )
    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=trainset, eval_dataset=evalset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.ps_encoder.requires_grad_(requires_grad=True)
    model.bytes_encoder.requires_grad_(requires_grad=True)

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=learning_rate, per_device_train_batch_size=batch_size, num_train_epochs=n_epochs,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='no',
        label_names=['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label'],
        bf16=True,
    )
    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=trainset, eval_dataset=evalset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_state()
    torch.save(model.state_dict(), os.path.join(model_dir,"pytorch_model.bin"))

    model.eval()
    y_pred = []
    for i in tqdm(range(0, testset.num_rows, batch_size)):
        batch_inputs = {k: v.to(device) for k, v in testset[i:i + batch_size].items()}
        y_pred_tmp = model(batch_inputs)['y_logit'].argmax(1).cpu().numpy()
        y_pred = np.concatenate([y_pred, y_pred_tmp])
    print(accuracy_score(y_true=testset['y_label'], y_pred=y_pred))
    print(confusion_matrix(y_true=testset['y_label'], y_pred=y_pred))
    print(classification_report(y_true=testset['y_label'], y_pred=y_pred, digits=4, target_names=labels))

    training_result = {
        'type': f"BERTmm-att-ps-bytes-{model_type}",
        'model_type': model_type,
        'accuracy': accuracy_score(y_true=testset['y_label'], y_pred=y_pred),
        'confusion_matrix': confusion_matrix(y_true=testset['y_label'], y_pred=y_pred).tolist(),
        'classification_report': classification_report(y_true=testset['y_label'], y_pred=y_pred, output_dict=True)
    }
    training_results.append(training_result)
    with open(os.path.join(model_root, timestamp, 'training_results.json'), 'w') as f:
        json.dump(training_results, f)
