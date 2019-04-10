#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
From MTO file, randomly sample M variants and get top N LODt score variants.
Then, store essential information of the variants as TSV file.

* Prerequisite
    1. Run 03_mto_summary.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp, eprint

import numpy as np
import os
import sys

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


class _Variant:
    def __init__(self):
        self.chrom = '.'
        self.pos = 0
        self.ref = '.'
        self.alt = '.'
        self.lodt = 0.0
        self.vaf = 0.0

    def __str__(self):
        return f'{self.chrom}\t{self.pos}\t{self.ref}\t{self.alt}\t{self.lodt}\t{self.vaf}'

    def parse_tsv_entry(self, tsv_entry):
        # TSV fields: chrom, pos, ref, alt, LODt score, VAF
        tsv_fields = tsv_entry.strip().split('\t')
        self.chrom = tsv_fields[0]
        self.pos = int(tsv_fields[1])
        self.ref = tsv_fields[2]
        self.alt = tsv_fields[3]
        self.lodt = float(tsv_fields[4])
        self.vaf = float(tsv_fields[5])

    @staticmethod
    def parse_tsv_file(tsv_file_path):
        variants = []

        with open(tsv_file_path, 'r') as tsv_file:
            tsv_file.readline()  # remove a header

            for tsv_entry in tsv_file:
                variant = _Variant()
                variant.parse_tsv_entry(tsv_entry)
                variants.append(variant)

        return variants


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    script = os.path.abspath(__file__)
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.Mutect.Out.Summary'
    job_name_prefix = 'Minu.Var.Sampling'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    m = 10000  # No. randomly sampled variants
    n = 1000  # No. top LODt score variants
    num_iter = 1000  # No. attempts of sampling
    cell_lines = ['HCC1143', 'HCC1954']
    depths = ['10x', '20x', '30x', '40x', '50x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for cell_line in cell_lines:
        for depth in depths:
            # path settings
            mto_summary_dir = f'{PROJECT_DIR}/results/mto-summary/{cell_line}/{depth}'
            var_sample_dir = f'{PROJECT_DIR}/results/variant-samples/{cell_line}/{depth}'

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                mto_summary_path = f'{mto_summary_dir}/{cell_line}.{purity_tag}.{depth}.tsv'  # input

                if not os.path.isfile(mto_summary_path):
                    continue

                out_dir = f'{var_sample_dir}/{purity_tag}'
                os.makedirs(out_dir, exist_ok=True)

                cmd = ''
                job_index = 1
                job_cnt_one_cmd = 40

                for i in range(num_iter):
                    out_tsv_path = f'{out_dir}/rand_{m}_top_{n}_{i+1:04}.tsv'
                    cmd += f'{script} make_rand_sample_var_file {out_tsv_path} {mto_summary_path} {m} {n};'

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


def make_rand_sample_var_file(out_tsv_path, in_tsv_path, num_rand_var, num_top_lod_var):
    """
    Parse the input TSV file and randomly sample variants.
    From the variant set, get variants with top LODt score and store essential information of the variants as TSV file.

    * Essential information: chrom, pos, ref, alt, LODt score, VAF

    :param out_tsv_path: a path of an output TSV file
    :param in_tsv_path: a path of a input TSV file
    :param num_rand_var: No. of randomly sampled variants
    :param num_top_lod_var: No. of variants with top LODt scores
    """
    num_rand_var = int(num_rand_var)
    num_top_lod_var = int(num_top_lod_var)

    eprint('[LOG] Parameters')
    eprint(f'[LOG] --- Random sample size: {num_rand_var}')
    eprint(f'[LOG] --- No. variants with top LODt scores: {num_top_lod_var}')
    eprint()

    variants = []

    with open(in_tsv_path, 'r') as in_tsv_file:
        header = in_tsv_file.readline().strip()

        for line in in_tsv_file:
            variant = _Variant()
            variant.parse_tsv_entry(line)
            variants.append(variant)

    var_cnt = len(variants)

    eprint('[LOG] Random sampling')
    if var_cnt >= num_rand_var:
        random_variants = list(np.random.choice(variants, num_rand_var, replace=False))
    else:
        random_variants = list(np.random.choice(variants, num_rand_var, replace=True))

    random_variants.sort(key=lambda x: x.lodt, reverse=True)  # sorted by their LODt score

    eprint(f'[LOG] Write the information of variants with top {num_top_lod_var} LODt')
    with open(out_tsv_path, 'w') as out_tsv_file:
        print(header, file=out_tsv_file)

        for i in range(num_top_lod_var):
            print(random_variants[i], file=out_tsv_file)


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
