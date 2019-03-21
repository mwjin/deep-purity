#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Execute Mutect (Cibulskis et al., 2013, Nat. Biotech.) to call the variants from our normal contaminated bam files.
Called variants are going to be used to make images for our CNN.

* Prerequisite
    1. Run bam_mixer.py
"""

from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import sys

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    This is a bootstrap.
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True
    job_name_prefix = 'Minu.Mutect'
    log_dir = '%s/log/%s/%s' % (PROJECT_DIR, job_name_prefix, time_stamp())

    # param settings
    cell_line = 'HCC1954'
    depth = '30x'

    # path settings
    # input
    in_bam_dir = '%s/results/mixed-bam' % PROJECT_DIR
    in_bam_path_format = '{0}/{1}.{2}.%s.bam'.format(in_bam_dir, cell_line, depth)
    norm_bam_path = '/extdata6/Beomman/raw-data/tcga-benchmark4/%s.NORMAL.%s.compare.bam' % (cell_line, depth)  # ctrl
    # output
    out_dir = '%s/results/mixed-bam-var-call' % PROJECT_DIR
    out_vcf_path_format = '{0}/{1}.{2}.%s.vcf'.format(out_dir, cell_line, depth)
    out_mto_path_format = '{0}/{1}.{2}.%s.mto'.format(out_dir, cell_line, depth)
    os.makedirs(out_dir, exist_ok=True)

    # constant paths for mutect
    java = '/usr/java/latest/bin/java'  # must be JDK1.7
    temp_dir = '%s/log' % out_dir  # error log files will be stored.
    mutect = '/extdata6/Beomman/bins/mutect/mutect-1.1.7.jar'
    ref_genome = '/extdata6/Minwoo/data/ref-genome/hg19/Homo_sapiens_assembly19.fasta'
    dbsnp_path = '/extdata6/Beomman/raw-data/dbsnp/dbsnp150/compressed/hg19/common_all_20170710.hg19.vcf.gz'

    jobs = []  # a list of the 'Job' class
    norm_contam_pcts = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for norm_pct in norm_contam_pcts:
        tumor_pct = 100 - norm_pct
        tag = 'n%st%s' % (norm_pct, tumor_pct)

        # in-loop path settings
        in_bam_path = in_bam_path_format % tag
        out_vcf_path = out_vcf_path_format % tag
        out_mto_path = out_mto_path_format % tag

        cmd = f"{java} -Xmx2g -Djava.io.tmpdir={temp_dir} -jar {mutect} -rf BadCigar --analysis_type MuTect " \
              f"--reference_sequence {ref_genome} --dbsnp {dbsnp_path} " \
              f"--input_file:normal {norm_bam_path} --input_file:tumor {in_bam_path} " \
              f"--tumor_lod 6.3 --initial_tumor_lod 4.0 --fraction_contamination 0.0 " \
              f"--vcf {out_vcf_path} -o {out_mto_path} --enable_extended_output"

        if is_test:
            print(cmd)
        else:
            one_job_name = '%s.%s.%s.%s' % (job_name_prefix, cell_line, depth, tag)
            one_job = Job(one_job_name, cmd)
            jobs.append(one_job)

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
            sys.exit('ERROR: function_name=%s, parameters=%s' % (function_name, function_parameters))
