#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Estimate tumor purity with results after finding local extrema of a KDE of a VAF histogram

* Prerequisite
    Run kde.py
"""

import os
import sys

from decimal import Decimal

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'
KDE_BANDWIDTH = sys.argv[1]  # kde parameter; the histogram is more smooth at low bandwidth.


def main():
    """
    Boostrap
    """
    # param settings
    cells = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    # path settings
    kde_result_dir = f'{PROJECT_DIR}/results/heuristic/kde-normal-filt/{KDE_BANDWIDTH}'  # input
    out_dir = f'{PROJECT_DIR}/results/heuristic/est-tumor-purity'
    os.makedirs(out_dir, exist_ok=True)

    for cell in cells:
        for depth in depths:
            print(f'[LOG] Cell line: {cell}, Depth: {depth}, KDE Bandwidth: {KDE_BANDWIDTH}', file=sys.stderr)
            # in-loop path settings
            lextrema_dir = f'{kde_result_dir}/{cell}/{depth}'
            outfile_path = f'{out_dir}/purity_{cell}_{depth}_{KDE_BANDWIDTH}.txt'

            with open(outfile_path, 'w') as outfile:
                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
                    lextrema_file_path = f'{lextrema_dir}/local_extreme_vaf_{purity_tag}_{depth}.txt'

                    if not os.path.isfile(lextrema_file_path):
                        print(f'[WARNING] \'{lextrema_file_path}\' does not exist.')
                        continue

                    tumor_purity_ratio = tumor_purity / 100
                    est_purity_ratio = estimate_tumor_purity(lextrema_file_path)
                    print(f'{tumor_purity_ratio}\t{est_purity_ratio:.3f}', file=outfile)


def estimate_tumor_purity(lextrema_file_path):
    """
    Estimate a tumor purity from a result of finding VAFs in local extrema (results of kde.py)
    """
    vaf_dict = {'lmax': [], 'lmin': []}  # value: a list of VAFs

    with open(lextrema_file_path, 'r') as lextrema_file:
        for line in lextrema_file:
            cols = line.strip().split('\t')
            vaf = Decimal(cols[0])
            lextrema_label = cols[1]  # lmax or lmin
            vaf_dict[lextrema_label].append(vaf)

    lmax_cnt = len(vaf_dict['lmax'])
    assert lmax_cnt == 2 or lmax_cnt == 1  # bimodal or unimodal

    if lmax_cnt == 1:  # unimodal
        est_purity = vaf_dict['lmax'][0] * 2
    else:
        vaf_dict['lmax'].sort()
        est_purity = ((vaf_dict['lmax'][0] * 2) + vaf_dict['lmax'][1]) / 2

    return float(est_purity)


if __name__ == '__main__':
    main()
