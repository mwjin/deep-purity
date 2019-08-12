#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make data to test using all variants in each cell line

* Prerequisite
    1. Run 01_segment_cell.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import sys

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True
    prev_job_prefix = 'Minu.DeepPurity.CHAT.Segments'
    job_name_prefix = 'Minu.DeepPurity.Make.Test.Data'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    data_maker_script = f'{PROJECT_DIR}/src/data_maker.py'
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv/original'  # input
    seg_tsv_dir = f'{PROJECT_DIR}/data/segments-tsv/original'  # input
    test_data_dir = f'{PROJECT_DIR}/data/test-data'  # output
    data_list_dir = f'{PROJECT_DIR}/data/test-data-list'  # output
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(data_list_dir, exist_ok=True)

    if not os.path.isdir(seg_tsv_dir):
        sys.exit(f'[ERROR] The directory \'{seg_tsv_dir}\' does not exist. '
                 f'Check the path or run evaluation/01_segment_cell.py')

    # path settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for cell_line in cell_lines:
        # in-loop path settings
        test_data_list_path = f'{data_list_dir}/{cell_line}_{depth}_data_paths.txt'
        in_var_tsv_dir = f'{var_tsv_dir}/{cell_line}/{depth}'
        in_seg_tsv_dir = f'{seg_tsv_dir}/{cell_line}/{depth}'
        out_test_data_dir = f'{test_data_dir}/{cell_line}/{depth}'
        os.makedirs(out_test_data_dir, exist_ok=True)

        test_data_paths = []

        for norm_contam in norm_contams:
            tumor_purity = 100 - norm_contam
            purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
            tumor_purity_ratio = tumor_purity / 100

            in_var_tsv_path = f'{in_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
            in_seg_tsv_path = f'{in_seg_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
            out_test_data_path = f'{out_test_data_dir}/{cell_line}.{purity_tag}.{depth}.hdf5'
            test_data_paths.append(out_test_data_path)

            if not os.path.isfile(in_seg_tsv_path):
                sys.exit(f'[ERROR] The file \'{in_seg_tsv_path}\' does not exist. '
                         f'Check the path or run evaluation/01_segment_cell.py')

            cmd = f'{data_maker_script} {out_test_data_path} {in_var_tsv_path} {in_seg_tsv_path} ' \
                  f'{tumor_purity_ratio};'

            if is_test:
                print(cmd)
            else:
                prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                jobs.append(one_job)

        # make a list of vaf_hists made by this script for training and testing the model
        with open(test_data_list_path, 'w') as test_data_list_file:
            for test_data_path in test_data_paths:
                print(test_data_path, file=test_data_list_file)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


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
