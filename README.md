# MM4flow

The repository of the paper **MM4flow: A Pre-trained Multi-modal Model for Versatile Network Traffic Analysis**, published in the 32nd ACM Conference on Computer and Communications Security ([CCS'25]()).

Authored by 
[Luming Yang](https://shangshu-lab.github.io), 
Lin Liu*, 
[Junjie Huang](https://jjhuangcs.github.io), 
[Zhuotao Liu](https://liuzhuotao.github.io),
[Shiyu Liang](https://jhc.sjtu.edu.cn/people/members/faculty/shiyu-liang.html),
Shaojing Fu*,
Yongjun Wang*.  

<!--The code related to this paper will be open sourced in the future. -->

## Citation
For any work related to the analysis of encrypted video traffic, welcome to please cite our paper as:
```
@inproceedings{Yang2025MM4flow,
  title={MM4flow: A Pre-trained Multi-modal Model for Versatile Network Traffic Analysis},
  author={Yang, Luming and Liu, Lin and Huang, JunJie and Liu, Zhuotao and Liang, Shiyu and Fu, Shaojing and Wang, Yongjun},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS'25)},
  year = {2025},
  pages = {1664–1678},
  numpages = {15},
  publisher = {Association for Computing Machinery},
  location = {Taipei, CHINA},
  doi = {10.1145/3719027.3744804},
}
```

## Traffic Parsing

Zeek is an open source network security monitoring tool. Installation: https://docs.zeek.org/en/master/install.html

Use the Zeek plug-in to parse the pcap file and generate Zeek logs. 
Among them, ps.log records the packet length sequence, and raw.log records the payload byte stream.

```shell
zeek -r pcap_file ps.zeek bytes.zeek --no-checksums
```

## Pre-processing

The zeek log files of the dataset should be organized as ```$root_dir/$label/$pcapfile/*.log.gz```. Then, we can extract packet length sequences and payload byte streams from the dataset.

```shell
python proj_get_dataset.py -D log_dir -o output_dir
```

Then, we split the dataset into training set, validation set, and test set.

## Pre-training

Pre-training relies on large-scale unlabeled data to train separate models for each of the two modalities. 
This stage enables effective representation learning of the uni-modal network traffic information.

- Pre-train BERT-bytes

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_BERT_bytes.py -D pretrain_data_dir -d 768 -nh 12 -l 12 -f 3072 -b batch_size -ev 10 -sv 5 -ep n_epochs -lr 5e-5
  ```

- Pre-train BERT-ps

  ```shell
  CUDA_VISIBLE_DEVICES=4,5,6,7 python pretrain_BERT_ps.py -D pretrain_data_dir -d 768 -nh 12 -l 12 -f 3072 -b batch_size -ev 10 -sv 5 -ep n_epochs -lr 5e-5
  ```

  The pre-trained model is saved in "model/{timestamp}" 

## Fine-tuning

We performs supervised fine-tuning (SFT) on a small set of labeled dataset regarding specific downstream tasks.
This stage integrates information from both modalities to enhance the model's performance.

```shell
python proj_classifier_trainer.py -T timestamp -D dataset_dir -o output -b batch_size -ev 10 -sv 5 -ep n_epochs -lr 5e-5
```

The fine-tuned model is saved in "model-classifier/{timestamp}"

## Analysis

```shell
python proj_classifier_test.py -D log_dir -m {classifier-timestamp} -b batch_size -o output.csv
```

The analysis result is saved in "output.csv"

