#!/usr/bin/env python3
"""
Make data for training or testing our deep learning model

* Arguments of this script
    1. out_data_path = a path of a VAF histogram (that will be stored using h5py)
    2. var_tsv_path = a path of a mto summary file (.tsv)
    3. tumor purity (optional; only used for learning): a ratio of tumor samples
"""

import numpy as np
import pandas as pd
import h5py
import sys


def main():
    # argument parsing
    out_data_path = sys.argv[1]
    var_tsv_path = sys.argv[2]
    tumor_purity = float(sys.argv[3]) if len(sys.argv) >= 3 else None  # used for training and evaluation

    # params
    top_lodt_frac = 0.1

    # make a VAF histogram
    print('[LOG] Make a VAF histogram using somatic mutations')
    variant_df = pd.read_table(var_tsv_path)
    variant_df = variant_df[variant_df['judgement'] == 'KEEP']  # select only somatic variants
    variant_df = variant_df.sort_values(by=['t_lod_fstar'], ascending=False).reset_index(drop=True)
    variant_cnt = int(len(variant_df.index) * top_lodt_frac)
    variant_df = variant_df[:variant_cnt]
    vaf_hist = np.zeros(101)  # each item represents the number of variants for each VAF (unit: 0.01)

    for _, variant in variant_df.iterrows():
        vaf = variant['tumor_f']
        vaf_hist[int(round(vaf, 2) * 100)] += 1

    vaf_hist = vaf_hist / variant_cnt  # normalization
    del variant_df

    # store the vaf histogram
    with h5py.File(out_data_path, 'w') as outfile:
        outfile.create_dataset('vaf_hist_array', data=vaf_hist)
        outfile.create_dataset('tumor_purity', data=tumor_purity)


if __name__ == "__main__":
    main()
