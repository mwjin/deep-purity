#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
"""
Make images by parsing tsv files of variant samples via SGE job scheduler

* Prerequisite
    1. Run 05_sample_variants.py
    2. Run 06_segment_samples.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import sys

# constants
PROJECT_DIR = '/extdata1/baeklab/minwoo/projects/deep-purity'


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True
    prev_job_prefix = 'Minu.DeepPurity.CHAT.Segmentation'
    job_name_prefix = 'Minu.DeepPurity.Make.Learning.Data'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    data_maker_script = f'{PROJECT_DIR}/src/data_maker.py'
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv'
    chat_tsv_dir = f'{PROJECT_DIR}/data/chat-tsv'
    learn_data_dir = f'{PROJECT_DIR}/data/learning-data-2'
    data_list_dir = f'{PROJECT_DIR}/data/learning-data-2-list'
    os.makedirs(data_list_dir, exist_ok=True)

    # param settings
    num_files = 1000  # No. attempts of sampling

    data_classes = ['train-set', 'valid-set']
    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = list(range(5, 100, 5))
    jobs = []  # a list of the 'Job' class

    for data_class in data_classes:
        data_list_path = f'{data_list_dir}/{data_class.replace("-", "_")}_data_paths.txt'
        learn_data_paths = []

        for cell_line in cell_lines:
            for depth in depths:
                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
                    tumor_purity_ratio = tumor_purity / 100

                    # in-loop path settings
                    var_sample_dir = f'{var_tsv_dir}/{data_class}-samples/{cell_line}/{depth}/{purity_tag}'
                    chat_sample_dir = f'{chat_tsv_dir}/{data_class}-samples/{cell_line}/{depth}/{purity_tag}'
                    output_dir = f'{learn_data_dir}/{data_class}/{cell_line}/{depth}/{purity_tag}'
                    os.makedirs(output_dir, exist_ok=True)

                    cmd = ''
                    job_index = 1
                    job_cnt_one_cmd = 100  # it must be a divisor of {num_iter}.

                    for i in range(num_files):
                        in_var_tsv_path = f'{var_sample_dir}/random_variants_{i+1:04}.tsv'
                        in_chat_tsv_path = f'{chat_sample_dir}/random_variants_{i+1:04}.tsv'
                        output_path = f'{output_dir}/random_variants_{i+1:04}.hdf5'
                        learn_data_paths.append(output_path)
                        cmd += f'{data_maker_script} {output_path} {in_var_tsv_path} {in_chat_tsv_path} ' \
                               f'{tumor_purity_ratio};'

                        if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                            if is_test:
                                print(cmd)
                            else:
                                prev_job_name = f'{prev_job_prefix}.*'
                                one_job_name = \
                                    f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}.{data_class}.{job_index}'
                                one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                                jobs.append(one_job)

                            cmd = ''  # reset
                            job_index += 1

        # make a list of vaf_hists made by this script for training and testing the model
        with open(data_list_path, 'w') as data_list_file:
            for learn_data_path in learn_data_paths:
                print(learn_data_path, file=data_list_file)

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
