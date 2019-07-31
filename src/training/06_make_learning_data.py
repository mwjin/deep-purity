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
    image_maker_script = f'{PROJECT_DIR}/src/image_maker.py'
    image_set_dir = f'{PROJECT_DIR}/image-set-test-3'
    os.makedirs(image_set_dir, exist_ok=True)

    # param settings
    hist_width = 1000
    hist_height = 100
    mode = 1  # 0: images of variants (from 03_summarize_mto.py), 1: images of variant samples (from 05_sample_variants.py)

    # for mode 1
    m = 1000  # No. randomly sampled variants
    num_iter = 1000  # No. attempts of sampling

    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for cell_line in cell_lines:
        for depth in depths:
            # in-loop path settings
            tumor_bam_dir = f'/extdata4/baeklab/minwoo/data/TCGA-HCC-MIX/{cell_line}/{depth}'
            norm_bam_path = f'/extdata4/baeklab/minwoo/data/TCGA-HCC/{cell_line}/{cell_line}.NORMAL.{depth}.bam'

            if mode == 0:  # make images of variants
                image_set_path = f'{image_set_dir}/{cell_line}_{depth}.txt'
                image_paths = []

                var_tsv_dir = f'{PROJECT_DIR}/results/mto-summary/{cell_line}/{depth}'  # a result of 03_summarize_mto.py
                out_image_dir = f'{PROJECT_DIR}/results/variant-images/{cell_line}/{depth}'
                os.makedirs(out_image_dir, exist_ok=True)

                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
                    norm_contam_ratio = norm_contam / 100

                    # in-loop path settings
                    tumor_bam_path = f'{tumor_bam_dir}/{cell_line}.{purity_tag}.{depth}.bam'
                    var_tsv_path = f'{var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'

                    if not os.path.isfile(var_tsv_path):
                        continue

                    var_image_path = f'{out_image_dir}/{cell_line}.{purity_tag}.{depth}.pkl'  # output
                    image_paths.append(var_image_path)

                    cmd = f'{image_maker_script} {in_tsv_path} {norm_contam_ratio} {tumor_bam_path} ' \
                          f'{norm_bam_path} {var_image_path} {hist_width} {hist_height};'

                    if is_test:
                        print(cmd)
                    else:
                        prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                        one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                        one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                        jobs.append(one_job)

            else:  # make images of random variants
                image_set_path = f'{image_set_dir}/random_{m}_{cell_line}_{depth}.txt'
                image_paths = []

                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'
                    tumor_purity_ratio = tumor_purity / 100

                    # in-loop path settings
                    tumor_bam_path = f'{tumor_bam_dir}/{cell_line}.{purity_tag}.{depth}.bam'
                    var_sample_dir = f'{PROJECT_DIR}/results/variant-samples/{m}/{cell_line}/{depth}/{purity_tag}'

                    if not os.path.isdir(var_sample_dir):
                        continue

                    out_image_dir = f'{PROJECT_DIR}/results/variant-samples-images-test/' \
                                    f'{m}/{cell_line}/{depth}/{purity_tag}'
                    os.makedirs(out_image_dir, exist_ok=True)

                    cmd = ''
                    job_index = 1
                    job_cnt_one_cmd = 8  # it must be a divisor of {num_iter}.

                    for i in range(num_iter):
                        in_tsv_path = f'{var_sample_dir}/rand_{m}_{i+1:04}.tsv'
                        out_image_path = f'{out_image_dir}/rand_{m}_{i+1:04}.pkl'
                        image_paths.append(out_image_path)

                        cmd += f'{image_maker_script} {out_image_path} {in_tsv_path} ' \
                               f'{tumor_bam_path} {norm_bam_path} {tumor_purity_ratio};'

                        if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                            if is_test:
                                print(cmd)
                            else:
                                prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                                one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}.Random.{job_index}'
                                one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                                jobs.append(one_job)

                            cmd = ''  # reset
                            job_index += 1

            # make a list of images made by this script for training and testing the model
            with open(image_set_path, 'w') as image_set_file:
                for image_path in image_paths:
                    print(image_path, file=image_set_file)

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
