#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Classify variants into training and validation set using the files of the summarized MuTect outputs
Training set: chromosome 1, 3, 5, ..., 21
Test set: chromosome 2, 4, 6, ..., 22

* Prerequisite
    1. Run 03_summarize_mto.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import pandas as pd
import sys

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    script = os.path.abspath(__file__)
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.Mutect.Out.Summary'
    job_name_prefix = 'Minu.DeepPurity.Classify.Variants'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1954']
    depths = ['30x']

    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent
    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # path settings
            var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv'
            in_var_tsv_dir = f'{var_tsv_dir}/original/{cell_line}/{depth}'  # input
            train_var_tsv_dir = f'{var_tsv_dir}/train-set/{cell_line}/{depth}'
            valid_var_tsv_dir = f'{var_tsv_dir}/valid-set/{cell_line}/{depth}'
            os.makedirs(train_var_tsv_dir, exist_ok=True)
            os.makedirs(valid_var_tsv_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                in_var_tsv_path = f'{in_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                train_tsv_path = f'{train_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                valid_tsv_path = f'{valid_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'

                if not os.path.isfile(in_var_tsv_path):
                    continue

                cmd = f'{script} write_train_valid_variants {train_tsv_path} {valid_tsv_path} {in_var_tsv_path}'

                if is_test:
                    print(cmd)
                else:
                    prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                    jobs.append(one_job)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def write_train_valid_variants(out_train_tsv_path, out_valid_tsv_path, in_var_tsv_path):
    """
    Classify variants into training set and test set and save each of them as .tsv format.
    Training set: chromosome 1, 3, 5, ..., 21
    Test set: chromosome 2, 4, 6, ..., 22
    """
    variant_df = pd.read_table(in_var_tsv_path)
    train_df = variant_df[variant_df['contig'] % 2 == 1].reset_index(drop=True)
    valid_df = variant_df[variant_df['contig'] % 2 == 0].reset_index(drop=True)
    train_df.to_csv(out_train_tsv_path, sep='\t', index=False)
    valid_df.to_csv(out_valid_tsv_path, sep='\t', index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        function_name = sys.argv[1]
        function_parameters = sys.argv[2:]

        if function_name in locals().keys():
            locals()[function_name](*function_parameters)
        else:
            sys.exit(f'[ERROR]: The function \"{function_name}\" is unavailable.')
