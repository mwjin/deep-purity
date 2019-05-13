"""
Make a summary for a prediction result
In the summary file,
    column 1: tumor purity
    column 2: predicted tumor purity (mean of predictions)
"""

import numpy as np
import sys

# path settings
pred_result_path = sys.argv[1]
out_path = pred_result_path.replace('.txt', '_summary.txt')

purity_to_preds = {}

# parse the input file
with open(pred_result_path, 'r') as pred_result_file:
    for line in pred_result_file:
        cols = line.strip().split('\t')
        tumor_purity = round(float(cols[1]), 3)
        pred_tumor_purity = round(float(cols[2]), 3)

        if not tumor_purity in purity_to_preds:
            purity_to_preds[tumor_purity] = []

        purity_to_preds[tumor_purity].append(pred_tumor_purity)

# write a summary
with open(out_path, 'w') as outfile:
    for purity in purity_to_preds:
        preds = purity_to_preds[purity]
        mean_pred = round(float(np.mean(preds)), 3)
        print(purity, mean_pred, sep='\t', file=outfile)
