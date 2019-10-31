#!/usr/bin/env python3
"""
From the variant sample files, extract only germline variants hetero in normal and segment the genome like CHAT
(Li and Li., Genome Biology, 2014)

* Prerequisite
    1. Run 05_sample_variants.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp
from settings import PROJECT_DIR

import os
import sys


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.DeepPurity.Variant.Sampling'
    job_name_prefix = 'Minu.DeepPurity.CHAT.Segmentation'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv'
    seg_tsv_dir = f'{PROJECT_DIR}/data/segments-tsv'
    seg_script = f'{PROJECT_DIR}/src/segmentation.R'

    # param settings
    num_sampling = 1000  # No. attempts of sampling
    segment_size = 100

    variant_classes = ['train-set', 'valid-set']
    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = list(range(5, 100, 5))

    jobs = []  # a list of the 'Job' class

    for variant_class in variant_classes:
        for cell_line in cell_lines:
            for depth in depths:
                # path settings
                in_var_tsv_dir = f'{var_tsv_dir}/{variant_class}-samples/{cell_line}/{depth}'
                out_seg_dir = f'{seg_tsv_dir}/{variant_class}-samples/{cell_line}/{depth}'

                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
                    os.makedirs(f'{out_seg_dir}/{purity_tag}', exist_ok=True)

                    cmd = ''
                    job_index = 1
                    job_cnt_one_cmd = 40

                    for i in range(num_sampling):
                        in_var_tsv_path = f'{in_var_tsv_dir}/{purity_tag}/random_variants_{i+1:04}.tsv'
                        output_path = f'{out_seg_dir}/{purity_tag}/random_variants_{i+1:04}.tsv'

                        cmd += f'{seg_script} {output_path} {in_var_tsv_path} {segment_size};'

                        if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                            if is_test:
                                print(cmd)
                            else:
                                prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}.{variant_class}.*'
                                one_job_name = \
                                    f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}.{variant_class}.{job_index}'
                                one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                                jobs.append(one_job)

                            cmd = ''  # reset
                            job_index += 1

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
