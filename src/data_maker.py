#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make data for training or testing our deep learning model

* Arguments of this script
    1. out_data_path = a path of a VAF histogram (that will be stored using h5py)
    2. var_tsv_path = a path of a mto summary file (.tsv)
    3. chat_tsv_path = a path of a file for segments from the file of var_tsv_path
    4. tumor purity (optional; only used for learning): a ratio of tumor samples
"""

import numpy as np
import pandas as pd
import h5py
import sys

# argument parsing
out_data_path = sys.argv[1]
var_tsv_path = sys.argv[2]
chat_tsv_path = sys.argv[3]
tumor_purity = float(sys.argv[4]) if len(sys.argv) >= 4 else None  # used for training and evaluation

# params
top_lodt_frac = 0.1

# make a VAF histogram
print('[LOG] Make a VAF histogram using somatic mutations')
variant_df = pd.read_table(var_tsv_path)
variant_df = variant_df[variant_df['judgement'] == 'KEEP']
variant_df = variant_df.sort_values(by=['t_lod_fstar'], ascending=False).reset_index(drop=True)
variant_cnt = int(len(variant_df.index) * top_lodt_frac)
variant_df = variant_df[:variant_cnt]
vaf_hist = np.zeros(101)  # each item represents the number of variants for each VAF (unit: 0.01)

for _, variant in variant_df.iterrows():
    vaf = variant['tumor_f']
    vaf_hist[int(round(vaf, 2) * 100)] += 1

vaf_hist = vaf_hist / variant_cnt  # normalization
del(variant_df)

# make a image for folded VAF-log2 LRR plot in CHAT
print('[LOG] Make a folded VAF-log2 LRR image using segments')
chat_result_df = pd.read_table(chat_tsv_path)
chat_result_df = chat_result_df.dropna()
chat_seg_cnt = 1000

if len(chat_result_df.index) >= chat_seg_cnt:
    chat_result_df = chat_result_df.sample(n=chat_seg_cnt, replace=False)
else:
    chat_result_df = chat_result_df.sample(n=chat_seg_cnt, replace=True)

# chat_result_df = chat_result_df.sort_values(by=['folded_VAF', 'log2-LRR', 'min_square_dist'], ascending=False)
vaf_lrr_image = chat_result_df.to_numpy()
vaf_lrr_image =  vaf_lrr_image[:, :, np.newaxis]  # expand the dimension
del(chat_result_df)

# store the vaf histogram
with h5py.File(out_data_path, 'w') as outfile:
    outfile.create_dataset('vaf_hist_array', data=vaf_hist)
    outfile.create_dataset('vaf_lrr_image', data=vaf_lrr_image)
    outfile.create_dataset('tumor_purity', data=tumor_purity)
