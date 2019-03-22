#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
From MTO file, extract and store essential information of the passed variants as TSV file (Variant summary file).

* Prerequisite
    1. Run exe_mutect.py
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

    def __str__(self):
        return f'{self.chrom}\t{self.pos}\t{self.ref}\t{self.alt}\t{self.lodt}\t{self.vaf}'

    @staticmethod
    def parse_mto_file(mto_path):
        """
        Parse the MTO file and make and return '_Variant' objects
        """
        regex_chr = re.compile('^[0-9]{1,2}|[XY]$')
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

            for mto_entry in mto_file:
                fields = mto_entry.strip().split('\t')
                chrom = fields[chrom_idx]
                judge = fields[judge_idx]

                if regex_chr.match(chrom) and judge == 'KEEP':
                    variant = _Variant()
                    variant.chrom = chrom
                    variant.pos = int(fields[pos_idx])
                    variant.ref = fields[ref_idx]
                    variant.alt = fields[alt_idx]
                    variant.lodt = float(fields[lodt_idx])
                    variant.vaf = float(fields[vaf_idx])
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

    prev_job_prefix = 'Minu.Mutect'
    job_name_prefix = 'Minu.Var.Filtering'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cell_line = 'HCC1954'
    depth = '30x'

    # path settings
    mto_dir = f'{PROJECT_DIR}/results/mixed-bam-var-call'
    mto_path_format = f'{mto_dir}/{cell_line}.{depth}.%s.mto'
    out_tsv_dir = f'{mto_dir}'
    out_tsv_format = f'{out_tsv_dir}/{cell_line}.{depth}.%s.tsv'
    os.makedirs(out_tsv_dir, exist_ok=True)

    jobs = []  # a list of the 'Job' class
    norm_contam_pcts = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for norm_pct in norm_contam_pcts:
        tumor_pct = 100 - norm_pct
        tag = f'n{norm_pct}t{tumor_pct}'

        # in-loop path settings
        mto_path = mto_path_format % tag
        out_tsv_path = out_tsv_format % tag

        cmd = f'{script} make_variant_summary {out_tsv_path} {mto_path}'

        if is_test:
            print(cmd)
        else:
            prev_job_name = '%s.%s.%s.%s' % (prev_job_prefix, cell_line, depth, tag)
            one_job_name = '%s.%s.%s.%s' % (job_name_prefix, cell_line, depth, tag)
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
    tsv_header = 'chrom\tpos\tref_allele\talt_allele\tlod_t_score\tvaf'

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
