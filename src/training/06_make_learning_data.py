#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make images by parsing tsv files of variant samples via SGE job scheduler

* Prerequisite
    1. Run 05_sample_variants.py
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
    prev_job_prefix = 'Minu.DeepPurity.Variant.Sampling'
    job_name_prefix = 'Minu.DeepPurity.Make.Learning.Data'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    data_maker_script = f'{PROJECT_DIR}/src/data_maker.py'
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv'
    learn_data_dir = f'{PROJECT_DIR}/data/learning-data'
    data_list_dir = f'{PROJECT_DIR}/data/learning-data-list'  # output
    os.makedirs(data_list_dir, exist_ok=True)

    # param settings
    m = 10000  # No. randomly sampled variants
    num_iter = 1000  # No. attempts of sampling

    data_classes = ['train-set', 'valid-set']
    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

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

                    if not os.path.isdir(var_sample_dir):
                        continue

                    output_dir = f'{learn_data_dir}/{data_class}/{cell_line}/{depth}/{purity_tag}'
                    os.makedirs(output_dir, exist_ok=True)

                    cmd = ''
                    job_index = 1
                    job_cnt_one_cmd = 8  # it must be a divisor of {num_iter}.

                    for i in range(num_iter):
                        in_tsv_path = f'{var_sample_dir}/rand_{m}_{i+1:04}.tsv'
                        out_vaf_hist_path = f'{output_dir}/rand_{m}_{i+1:04}.hdf5'
                        learn_data_paths.append(out_vaf_hist_path)
                        cmd += f'{data_maker_script} {out_vaf_hist_path} {in_tsv_path} {tumor_purity_ratio};'

                        if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                            if is_test:
                                print(cmd)
                            else:
                                prev_job_name = f'{prev_job_prefix}.{data_class}.{cell_line}.{depth}.{purity_tag}'
                                one_job_name = f'{job_name_prefix}.{data_class}.{cell_line}.{depth}.{purity_tag}.' \
                                               f'Random.{job_index}'
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
