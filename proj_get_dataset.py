from zat.log_to_dataframe import LogToDataFrame
from tqdm import tqdm
import pandas as pd
import os
import re
import json
import argparse

info = ['ts','id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service']
conn_features = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'conn_state']
ps_features = ['up', 'down', 'ps']
raw_features = ['fwd_raw', 'bwd_raw']

log2df = LogToDataFrame()



min_pkts = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '-D', type=str, default='log_dir')
    parser.add_argument('--output', '-o', type=str, default='output')

    args = parser.parse_args()
    log_dir = args.log_dir
    output = args.output

    if not os.path.exists(output):
        os.mkdir(output)

    df = pd.DataFrame()
    for label in os.listdir(log_dir):
        print(label)
        # for filename in tqdm(os.listdir(os.path.join(rootdir, label))):
        for filename in os.listdir(os.path.join(log_dir, label)):
        #     print('\t', filename, end='\t')
            log_path = os.path.join(log_dir, label, filename)
            conn = log2df.create_dataframe(os.path.join(log_path, 'conn.log'), ts_index=False).set_index('uid')
            ps = log2df.create_dataframe(os.path.join(log_path, 'ps.log'), ts_index=False).set_index('uid')
            raw = log2df.create_dataframe(os.path.join(log_path, 'raw.log'), ts_index=False).set_index('uid')
            index = list(set(conn.index) & set(ps.index) & set(raw.index))
            df_tmp = pd.concat(
                [conn.loc[index][info + conn_features], ps.loc[index][ps_features], raw.loc[index][raw_features]],
                axis=1)
            df_tmp['label'] = label
            df = pd.concat([df, df_tmp])
    print(pd.DataFrame([df['label'].value_counts(), df[df['up']+df['down']>=min_pkts]['label'].value_counts()]).T)
    df.to_csv(os.path.join(output, "dataset.csv.gz"), compression='gzip')

