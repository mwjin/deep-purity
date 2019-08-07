#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
From MTO file, randomly sample variants, then store essential information of the variants as TSV file.
The TSV files will be used as data sets for learning

* Prerequisite
    1. Run 04_classify_variants.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp, eprint

import pandas as pd
import os
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

    prev_job_prefix = 'Minu.DeepPurity.Classify.Variants'
    job_name_prefix = 'Minu.DeepPurity.Variant.Sampling'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv'

    # param settings
    num_rand_somatic = 10000  # No. randomly sampled somatic variants
    num_rand_germline = 150000  # No. randomly sampled germline variants heterozygous in normal
    num_sampling = 1000  # No. attempts of sampling

    variant_classes = ['train-set', 'valid-set']
    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for variant_class in variant_classes:
        for cell_line in cell_lines:
            for depth in depths:
                # path settings
                in_var_tsv_dir = f'{var_tsv_dir}/{variant_class}/{cell_line}/{depth}'
                out_var_tsv_dir = f'{var_tsv_dir}/{variant_class}-samples/{cell_line}/{depth}'

                for norm_contam in norm_contams:
                    tumor_purity = 100 - norm_contam
                    purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                    # in-loop path settings
                    in_var_tsv_path = f'{in_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'  # input

                    if not os.path.isfile(in_var_tsv_path):
                        continue

                    out_sample_dir = f'{out_var_tsv_dir}/{purity_tag}'
                    os.makedirs(out_sample_dir, exist_ok=True)

                    cmd = ''
                    job_index = 1
                    job_cnt_one_cmd = 40

                    for i in range(num_sampling):
                        out_tsv_path = f'{out_sample_dir}/random_variants_{i+1:04}.tsv'
                        cmd += f'{script} write_variant_sample {out_tsv_path} {in_var_tsv_path} ' \
                               f'{num_rand_somatic} {num_rand_germline};'

                        if i % job_cnt_one_cmd == job_cnt_one_cmd - 1:
                            if is_test:
                                print(cmd)
                            else:
                                prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                                one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}.{job_index}'
                                one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                                jobs.append(one_job)

                            cmd = ''  # reset
                            job_index += 1

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def write_variant_sample(out_tsv_path, in_tsv_path, num_rand_somatic, num_rand_germline):
    """
    Parse the input TSV file and randomly sample variants.
    From the variant set, get variants with top LODt score and store essential information of the variants as TSV file.

    * Essential information: chrom, pos, ref, alt, LODt score, VAF

    :param out_tsv_path: a path of an output TSV file
    :param in_tsv_path: a path of a input TSV file
    :param num_rand_somatic: No. of randomly sampled somatic variants
    :param num_rand_germline: No. of randomly sampled germline variants heterozygous in normal
    """
    num_rand_somatic = int(num_rand_somatic)
    num_rand_germline = int(num_rand_germline)

    eprint('[LOG] Parameters')
    eprint(f'[LOG] --- No. random somatic mutations: {num_rand_somatic}')
    eprint(f'[LOG] --- No. germline somatic mutations: {num_rand_germline}')

    variant_df = pd.read_table(in_tsv_path)
    somatic_variant_df = variant_df[variant_df['judgement'] == 'KEEP']
    germline_variant_df = variant_df[variant_df['judgement'] == 'REJECT']

    eprint('[LOG] Random sampling')
    somatic_var_cnt = len(somatic_variant_df.index)
    germline_var_cnt = len(germline_variant_df.index)

    if somatic_var_cnt >= num_rand_somatic:
        somatic_variant_df = somatic_variant_df.sample(n=num_rand_somatic, replace=False)
    else:
        somatic_variant_df = somatic_variant_df.sample(n=num_rand_somatic, replace=True)

    if germline_var_cnt >= num_rand_germline:
        germline_variant_df = germline_variant_df.sample(n=num_rand_germline, replace=False)
    else:
        germline_variant_df = germline_variant_df.sample(n=num_rand_germline, replace=True)

    variant_df = pd.concat([somatic_variant_df.reset_index(drop=True),
                            germline_variant_df.reset_index(drop=True)],
                           axis=0)
    variant_df = variant_df.sort_values(by=['contig', 'position'])
    variant_df.to_csv(out_tsv_path, sep='\t', index=False)


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
