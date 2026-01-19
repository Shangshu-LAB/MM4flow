from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import torch
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
import itertools
import random



import argparse

min_pkts = 5
max_length = 512


tokenizer_bytes = BertTokenizerFast.from_pretrained('tokenizer_bert/bytes_tokenizer')

model_name = "BERT-bytes"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='log_sample')

    parser.add_argument('--d_model', '-d', type=int, default=128)  # 32
    parser.add_argument('--n_head', '-nh', type=int, default=2)  # 4
    parser.add_argument('--n_layer', '-l', type=int, default=2)  # 4
    parser.add_argument('--dim_ff', '-f', type=int, default=512)  # 64

    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--eval_per_epoch', '-ev', type=int, default=100)
    parser.add_argument('--save_per_epoch', '-sv', type=int, default=5)
    parser.add_argument('--n_epochs', '-ep', type=int, default=2)
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5)

    args = parser.parse_args()

    dataset_path = args.dataset

    d_model = args.d_model
    n_head = args.n_head
    dim_ff = args.dim_ff
    n_layer = args.n_layer

    batch_size = args.batch_size
    eval_per_epoch = args.eval_per_epoch
    save_per_epoch = args.save_per_epoch
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"There are {gpu_count} GPU is available.")
    else:
        gpu_count = 1
        print("No GPU is available.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_path = os.path.join("model", f"{model_name}_{timestamp}")
    print(model_path)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    def encode(examples):
        fwd_raw = [' '.join([raw[i:i + 2] for i in range(0, len(raw), 2)]) for raw in examples['fwd_raw']]
        bwd_raw = [' '.join([raw[i:i + 2] for i in range(0, len(raw), 2)]) for raw in examples['bwd_raw']]
        raw = tokenizer_bytes(
            list(zip(fwd_raw, bwd_raw)),
            truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True
        )
        return raw


    hyperparameters = {
        'd_model': d_model, 'n_head': n_head, 'dim_ff': dim_ff, 'n_layer': n_layer,
        'max_length': max_length,
        'dataset': dataset_path,
    }

    trainset_filepaths = [os.path.join(f"{dataset_path}_trainset__", filename) for filename in os.listdir(f"{dataset_path}_trainset__") if filename.split('.')[-1] == 'gz']
    print(trainset_filepaths)
    trainset = load_dataset('csv', data_files=trainset_filepaths, streaming=True)
    trainset = trainset.map(encode, batched=True)
    trainset = trainset.shuffle()

    with open(os.path.join(f"{dataset_path}_trainset__", "info.json")) as f:
        trainset_info = json.load(f)
    trainset_nrows = trainset_info['nrow']

    testset_filepaths = [os.path.join(f"{dataset_path}_testset__", filename) for filename in os.listdir(f"{dataset_path}_testset__") if filename.split('.')[-1] == 'gz']
    print(testset_filepaths)
    testset = load_dataset('csv', data_files=testset_filepaths, streaming=True)
    testset = testset.map(encode, batched=True)
    testset = testset.shuffle()


    model_config = BertConfig(
        vocab_size=len(tokenizer_bytes.get_vocab()), max_position_embeddings=max_length,
        hidden_size=hyperparameters['d_model'], # 256/512/768/1024
        num_hidden_layers=hyperparameters['n_layer'], # 3/6/12/24
        num_attention_heads=hyperparameters['n_head'], # 4/8/12/16,
        intermediate_size=hyperparameters['dim_ff'], # 3072
    )
    model = BertForMaskedLM(config=model_config)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"There are {model_size/(1000*1000):.2f}M parameters in the model.")

    hyperparameters['model_size'] = model_size
    with open(os.path.join(model_path,'hyperparameters.json'), 'w') as func:
        json.dump(hyperparameters, func)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_bytes, mlm=True, mlm_probability=0.2
    )
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        learning_rate=learning_rate,

        # dataloader_num_workers=max(len(trainset_filepaths), len(testset_filepaths)),
        bf16=True,

        per_device_train_batch_size=batch_size,
        max_steps=int(trainset_nrows / (batch_size * gpu_count)) * n_epochs,
        # num_train_epochs=n_epochs,

        logging_dir=os.path.join(model_path, 'logs'),
        # logging_strategy='epoch',
        logging_strategy='steps',
        logging_steps=int(trainset_nrows / (batch_size * gpu_count * eval_per_epoch)),
        # logging_steps=100,

        per_device_eval_batch_size=batch_size,
        evaluation_strategy='steps',
        eval_steps=int(trainset_nrows / (batch_size * gpu_count * eval_per_epoch)),
        # eval_steps=100,

        save_strategy='steps',
        save_steps = int(trainset_nrows / (batch_size * gpu_count * save_per_epoch)),

        # fsdp=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator,
        train_dataset=trainset['train'].with_format('torch'),
        eval_dataset=testset['train'].with_format('torch'),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    trainer.save_state()