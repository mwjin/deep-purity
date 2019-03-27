#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make images by parsing tsv files of variant samples via SGE job scheduler

* Prerequisite
    1. Run preprocess.04_make_sample.py
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
    is_test = False
    prev_job_prefix = 'Minu.Var.Sampling'
    job_name_prefix = 'Minu.Make.Var.Image'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    m = 10000  # No. randomly sampled variants
    n = 1000  # No. top LODt score variants
    num_iter = 1000  # No. attempts of sampling

    cell_line = 'HCC1143'
    depth = '30x'

    hist_width = 1000
    hist_height = 100

    # path settings
    bam_dir = '/extdata6/Beomman/raw-data/tcga-benchmark4'
    tumor_bam_path_format = f'{bam_dir}/{cell_line}.TUMOR.{depth}.compare.bam_%s.bam'
    norm_bam_path = f'{bam_dir}/{cell_line}.NORMAL.{depth}.compare.bam'

    image_maker_script = f'{PROJECT_DIR}/codes/MakeImage.py'
    sample_tsv_dir_format = f'{PROJECT_DIR}/results/var-samples/{cell_line}/{depth}/%s'
    out_image_dir_format = f'{PROJECT_DIR}/results/var-sample-images/{cell_line}/{depth}/%s'

    image_set_dir = f'{PROJECT_DIR}/image-set'
    image_set_path = f'{image_set_dir}/train_{cell_line}_{depth}.txt'
    image_path_list = []
    os.makedirs(image_set_dir, exist_ok=True)

    # make jobs
    jobs = []  # a list of the 'Job' class
    norm_contam_pcts = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for norm_pct in norm_contam_pcts:
        tumor_pct = 100 - norm_pct
        tag = f'n{norm_pct}t{tumor_pct}'
        norm_contam_ratio = norm_pct / 100

        # in-loop path settings
        tumor_bam_path = tumor_bam_path_format % f'n{int(norm_pct)}t{int(tumor_pct)}'
        sample_tsv_dir = sample_tsv_dir_format % tag
        out_image_dir = out_image_dir_format % tag
        os.makedirs(out_image_dir, exist_ok=True)

        cmd = ''
        job_index = 1
        job_cnt_one_cmd = 5

        for i in range(num_iter):
            in_tsv_path = f'{sample_tsv_dir}/rand_{m}_top_{n}_{i+1:04}.tsv'
            out_image_path = f'{out_image_dir}/rand_{m}_top_{n}_{i+1:04}.pkl'
            image_path_list.append(out_image_path)

            cmd += f'{image_maker_script} {in_tsv_path} {norm_contam_ratio} {tumor_bam_path} {norm_bam_path} ' \
                   f'{out_image_path} {hist_width} {hist_height};'

            if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                if is_test:
                    print(cmd)
                else:
                    prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{tag}'
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{tag}.{job_index}'
                    one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                    jobs.append(one_job)

                cmd = ''  # reset
                job_index += 1

    if not is_test:
        qsub_sge(jobs, queue, log_dir)

    # make a list of images made by this script for training and testing the model
    with open(image_set_path, 'w') as image_set_file:
        for image_path in image_path_list:
            print(image_path, file=image_set_file)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        function_name = sys.argv[1]
        function_parameters = sys.argv[2:]

        if function_name in locals().keys():
            locals()[function_name](*function_parameters)
        else:
            sys.exit('ERROR: function_name=%s, parameters=%s' % (function_name, function_parameters))
