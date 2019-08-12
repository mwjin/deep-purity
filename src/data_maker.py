#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make data for training or testing our deep learning model

* Arguments of this script
    1. out_data_path = a path of a VAF histogram (that will be stored using h5py)
    2. var_tsv_path = a path of a mto summary file (.tsv)
    3. seg_tsv_path = a path of a file for segments from the file of var_tsv_path
    4. tumor purity (optional; only used for learning): a ratio of tumor samples
"""

import numpy as np
import pandas as pd
import h5py
import sys

# argument parsing
out_data_path = sys.argv[1]
var_tsv_path = sys.argv[2]
seg_tsv_path = sys.argv[3]
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

# make a image for folded VAF-log2 LRR plot in CHAT
print('[LOG] Make a folded VAF-log2 LRR image using segments')
segment_df = pd.read_table(seg_tsv_path)
segment_df = segment_df.dropna()
vaf_lrr_image = np.zeros((401, 501))

for _, segment in segment_df.iterrows():
    folded_vaf = segment['folded_VAF']
    log2_lrr = segment['log2-LRR']
    row_idx = int(round(log2_lrr, 2) * 100) + 200
    col_idx = int(round(folded_vaf, 3) * 1000)

    if row_idx < 0 or row_idx > 400 or col_idx > 500:
        continue

    vaf_lrr_image[row_idx, col_idx] = 1.0

vaf_lrr_image =  vaf_lrr_image[:, :, np.newaxis]  # expand the dimension

# store the vaf histogram
with h5py.File(out_data_path, 'w') as outfile:
    outfile.create_dataset('vaf_hist_array', data=vaf_hist)
    outfile.create_dataset('vaf_lrr_image', data=vaf_lrr_image)
    outfile.create_dataset('tumor_purity', data=tumor_purity)
