#!/usr/bin/env python3
"""
From MTO (MuTect output) file, extract and store essential information of the passed variants as .tsv file.
This is for reducing memory overhead.
"""
import re
import sys


def write_mto_summary(out_tsv_path, in_mto_path):
    """
    Parse a MTO file and write essential information of the passed variants as TSV file.

    * Essential information: chrom, pos, ref, alt, LODt score, VAF

    :param out_tsv_path: a path of an output (TSV)
    :param in_mto_path: a path of a MTO input file
    """
    # parse mto
    variants = _Variant.parse_mto_file(in_mto_path)
    tsv_header = 'contig\tposition\tref_allele\talt_allele\tt_lod_fstar\ttumor_f\t' \
                 't_ref_count\tt_alt_count\tn_ref_count\tn_alt_count\t' \
                 'judgement\tfailure_reasons'

    with open(out_tsv_path, 'w') as out_tsv_file:
        print(tsv_header, file=out_tsv_file)

        for variant in variants:
            print(variant, file=out_tsv_file)


class _Variant:
    def __init__(self):
        self.chrom = '.'
        self.pos = 0
        self.ref = '.'
        self.alt = '.'
        self.lodt = 0.0  # t_lod_fstar
        self.vaf = 0.0  # tumor_f
        self.t_ref_count = 0
        self.t_alt_count = 0
        self.n_ref_count = 0
        self.n_alt_count = 0
        self.judge = ''
        self.fail_reason = ''

    def __str__(self):
        return f'{self.chrom}\t{self.pos}\t{self.ref}\t{self.alt}\t{self.lodt}\t{self.vaf}\t' \
               f'{self.t_ref_count}\t{self.t_alt_count}\t{self.n_ref_count}\t{self.n_alt_count}\t' \
               f'{self.judge}\t{self.fail_reason}'

    @staticmethod
    def parse_mto_file(mto_path):
        """
        Parse the MTO file and make and return '_Variant' objects
        """
        regex_chr = re.compile('^(chr)?([0-9]{1,2})$')
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
            t_ref_count_idx = header_fields.index('t_ref_count')
            t_alt_count_idx = header_fields.index('t_alt_count')
            n_ref_count_idx = header_fields.index('n_ref_count')
            n_alt_count_idx = header_fields.index('n_alt_count')
            judge_idx = header_fields.index('judgement')
            fail_reason_idx = header_fields.index('failure_reasons')

            for mto_entry in mto_file:
                cols = mto_entry.strip().split('\t')
                chrom = cols[chrom_idx]
                judge = cols[judge_idx]
                fail_reason = cols[fail_reason_idx]

                if regex_chr.match(chrom):
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
                    variant.judge = judge
                    variant.fail_reason = fail_reason

                    # somatic mutations
                    if judge == 'KEEP':
                        variants.append(variant)
                        continue

        return variants


if __name__ == '__main__':
    write_mto_summary(out_tsv_path=sys.argv[1], in_mto_path=sys.argv[2])
