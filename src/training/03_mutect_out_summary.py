#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
From MTO (Mutect output) file, extract and store essential information of the passed variants as TSV file.
This is for reducing memory overhead.

* Prerequisite
    1. Run 02_run_mutect.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import re
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
        self.t_ref_count = 0
        self.t_alt_count = 0
        self.t_alt_freq = 0.0
        self.n_ref_count = 0
        self.n_alt_count = 0
        self.n_alt_freq = 0.0
        self.judge = ''

    def __str__(self):
        return f'{self.chrom}\t{self.pos}\t{self.ref}\t{self.alt}\t{self.lodt}\t{self.vaf}\t' \
               f'{self.t_ref_count}\t{self.t_alt_count}\t{self.t_alt_freq}\t' \
               f'{self.n_ref_count}\t{self.n_alt_count}\t{self.n_alt_freq}\t{self.judge}'

    @staticmethod
    def parse_mto_file(mto_path):
        """
        Parse the MTO file and make and return '_Variant' objects
        """
        regex_chr = re.compile('^(chr)?([0-9]{1,2}|[XY])$')
        variants = []

        with open(mto_path, 'r') as mto_file:
            mto_file.readline()
            header = mto_file.readline()
            header_fields = header.strip().split('\t')

            chrom_idx = header_fields.index('contig')
            pos_idx = header_fields.index('position')
            ref_idx = header_fields.index('ref_allele')
            alt_idx = header_fields.index('alt_allele')
            lodt_idx = header_fields.index('t_lod_fstar')
            vaf_idx = header_fields.index('tumor_f')
            judge_idx = header_fields.index('judgement')

            t_ref_count_idx = header_fields.index('t_ref_count')
            t_alt_count_idx = header_fields.index('t_alt_count')
            n_ref_count_idx = header_fields.index('n_ref_count')
            n_alt_count_idx = header_fields.index('n_alt_count')

            for mto_entry in mto_file:
                cols = mto_entry.strip().split('\t')
                chrom = cols[chrom_idx]
                judge = cols[judge_idx]

                if regex_chr.match(chrom) and judge == 'KEEP':
                    variant = _Variant()
                    variant.chrom = chrom
                    variant.pos = int(cols[pos_idx])
                    variant.ref = cols[ref_idx]
                    variant.alt = cols[alt_idx]
                    variant.lodt = float(cols[lodt_idx])
                    variant.vaf = float(cols[vaf_idx])

                    variant.t_ref_count = int(cols[t_ref_count_idx])
                    variant.t_alt_count = int(cols[t_alt_count_idx])
                    variant.n_ref_count = int(cols[n_ref_count_idx])
                    variant.n_alt_count = int(cols[n_alt_count_idx])

                    variant.t_alt_freq = variant.t_alt_count / (variant.t_ref_count + variant.t_alt_count)
                    variant.n_alt_freq = variant.n_alt_count / (variant.n_ref_count + variant.n_alt_count)
                    variant.judge = judge
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

    prev_job_prefix = 'Minu.Mutect.Variant.Call'
    job_name_prefix = 'Minu.Mutect.Out.Summary'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depths = ['30x']

    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # path settings
            mto_dir = f'{PROJECT_DIR}/results/mutect-output-depth-norm/{cell_line}/{depth}'
            mto_path_format = f'{mto_dir}/{cell_line}.%s.{depth}.mto'
            mto_summary_dir = f'{PROJECT_DIR}/results/mto-summary-depth-norm/{cell_line}/{depth}'
            mto_summary_path_format = f'{mto_summary_dir}/{cell_line}.%s.{depth}.tsv'
            os.makedirs(mto_summary_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                mto_path = mto_path_format % purity_tag
                out_tsv_path = mto_summary_path_format % purity_tag

                cmd = f'{script} make_variant_summary {out_tsv_path} {mto_path}'

                if is_test:
                    print(cmd)
                else:
                    prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                    jobs.append(one_job)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def make_variant_summary(out_tsv_path, in_mto_path):
    """
    Parse a MTO file and write essential information of the passed variants as TSV file.

    * Essential information: chrom, pos, ref, alt, LODt score, VAF

    :param out_tsv_path: a path of an output (TSV)
    :param in_mto_path: a path of a MTO input file
    """
    # parse mto
    variants = _Variant.parse_mto_file(in_mto_path)
    tsv_header = 'chrom\tpos\tref_allele\talt_allele\tlod_t_score\tvaf\t' \
                 't_ref_count\tt_alt_count\tt_alt_freq\t' \
                 'n_ref_count\tn_alt_count\tn_alt_freq\tjudgement'

    with open(out_tsv_path, 'w') as out_tsv_file:
        print(tsv_header, file=out_tsv_file)

        for variant in variants:
            print(variant, file=out_tsv_file)


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
