#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Kolmogorov-Smirnov (KS) test to choose a top LODt ratio and filter variants to remove noise
"""
import numpy
import scipy.stats
import sys
import os

from lab.job import Job, qsub_sge
from lab.utils import time_stamp

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
        # TSV cols: chrom, pos, ref, alt, LODt score, VAF
        tsv_cols = tsv_entry.strip().split('\t')
        self.chrom = tsv_cols[0]
        self.pos = int(tsv_cols[1])
        self.ref = tsv_cols[2]
        self.alt = tsv_cols[3]
        self.lodt = float(tsv_cols[4])
        self.vaf = float(tsv_cols[5])

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

    job_name_prefix = 'Minu.Purity.Heurisic.VAF.KS-Test.and.Filtering'
    log_dir = f'{PROJECT_DIR}/log/heuristic/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1187', 'HCC1954', 'HCC2218']
    depths = ['10x', '20x', '30x', '40x', '50x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # path settings
            var_tsv_dir = f'{PROJECT_DIR}/results/mto-summary/{cell_line}/{depth}'
            ks_result_dir = f'{PROJECT_DIR}/results/heuristic/ks-test/{cell_line}/{depth}'
            os.makedirs(ks_result_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                var_tsv_path = f'{var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                out_var_tsv_path = f'{ks_result_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                ks_result_path = f'{ks_result_dir}/ks_test_result_{purity_tag}.txt'

                cmd = f'{script} vaf_ks_test {out_var_tsv_path} {ks_result_path} {var_tsv_path};'

                if is_test:
                    print(cmd)
                else:
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job = Job(one_job_name, cmd)
                    jobs.append(one_job)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def vaf_ks_test(out_var_tsv_path, ks_result_path, var_tsv_path):
    """
    The function does KS test for VAFs to remove noise.
    For 20 top LODt ratios (0.05~1.00), the function does K-S test
    between VAFs of upper LODt variants and VAFs of lower LODt variants.
    The LODt ratio which has the lowest P value and where the difference of VAF median is more than 0.1 is selected.
    If there is at least one zero p-value, VAFs are down-sampled by 50%.
    """
    variants = _Variant.parse_tsv_file(var_tsv_path)
    variants.sort(key=lambda x: x.lodt, reverse=True)

    top_lodt_ratios = [round(0.05 * (x + 1), 2) for x in range(20)]
    ks_results = []  # item: (KS statistic, p value, LODt threshold)

    down_sample_variants = variants

    while True:  # if p-values != 0 for all KS tests, while loop will be broken.
        no_zero_pval = True

        for top_lodt_ratio in top_lodt_ratios:  # below code does ks test for each LODt threshold.
            boundary_idx = int(len(down_sample_variants) * top_lodt_ratio)

            # VAFs of the upper LODt group
            upper_sample_vafs = [x.vaf for x in down_sample_variants[:boundary_idx]]
            # VAFs of the lower LODt group
            lower_sample_vafs = [x.vaf for x in down_sample_variants[boundary_idx:]]

            stats, pval = scipy.stats.ks_2samp(upper_sample_vafs, lower_sample_vafs)  # k-s test
            ks_results.append((stats, pval, top_lodt_ratio))

            # if p value is 0, the code reset the ks test result and down sample the variant list by 50%.
            if pval == 0:
                down_sample_cnt = len(down_sample_variants) // 2
                down_sample_variants = list(numpy.random.choice(down_sample_variants, down_sample_cnt, replace=False))
                down_sample_variants.sort(key=lambda x: x.lodt, reverse=True)  # sorted by LODt scores
                no_zero_pval = False
                ks_results = []
                break

        if no_zero_pval:
            break

    ks_results.sort(key=lambda x: x[1])  # sort by p value
    top_lodt_ratio_cutoff = ks_results[0][2]
    upper_var_cnt = int(len(variants) * top_lodt_ratio_cutoff)

    upper_vafs = [x.vaf for x in variants[:upper_var_cnt]]
    lower_vafs = [x.vaf for x in variants[upper_var_cnt:]]

    # confirm the difference b/w median of upper VAFs and lower VAFs.
    upper_vaf_median = numpy.median(upper_vafs)
    lower_vaf_median = numpy.median(lower_vafs)
    diff_median = upper_vaf_median - lower_vaf_median

    # write the information of filtered variants
    with open(out_var_tsv_path, 'w') as outfile:
        if diff_median < 0.1:
            var_cnt = len(variants)
        else:
            var_cnt = upper_var_cnt

        tsv_header = 'chrom\tpos\tref_allele\talt_allele\tlod_t_score\tvaf'
        print(tsv_header, file=outfile)

        for i in range(var_cnt):
            print(variants[i], file=outfile)

    # write the results of KS test
    with open(ks_result_path, 'w') as ks_result_file:
        ks_result_file.write('KS_stat\tp-value\ttop_LODt_ratio\tdiff_median\n')
        ks_result_file.write('%s\t%s\t%s\t%s\n' % (ks_results[0][0], ks_results[0][1], ks_results[0][2], diff_median))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        function_name = sys.argv[1]
        function_parameters = sys.argv[2:]

        if function_name in locals().keys():
            locals()[function_name](*function_parameters)
        else:
            sys.exit(f'[ERROR] There is no \'{function_name}\' function.')
