#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
This script is used to mix normal reads (a%) and tumor reads ((1 - a)%) then make virtual tumor data in silico.
(a: normal contamination ratio)
"""
from random import randint
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import sys

# constants
SEED = randint(0, 1000000)
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    This is a bootstrap.
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True
    job_name_prefix = 'Minu.In.Silico.Bam.Mix'
    log_dir = '%s/log/%s/%s' % (PROJECT_DIR, job_name_prefix, time_stamp())

    # param settings
    cell_line = 'HCC1954'
    depth = '30x'

    # path settings
    samtools = '/extdata6/Doyeon/anaconda3/bin/samtools'
    java = '/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/java'
    picard = '/extdata6/Beomman/bins/picard/build/libs/picard.jar'
    tumor_bam_path = '/extdata6/Beomman/raw-data/tcga-benchmark4/%s.TUMOR.%s.compare.bam' % (cell_line, depth)
    norm_bam_path = '/extdata6/Beomman/raw-data/tcga-benchmark4/%s.NORMAL.%s.compare.bam' % (cell_line, depth)

    # output settings
    result_dir = '%s/results/mixed-bam' % PROJECT_DIR
    result_bam_path_format = '{0}/{1}.{2}.%s.bam'.format(result_dir, cell_line, depth)
    temp_dir = '%s/temp' % result_dir
    os.makedirs(temp_dir, exist_ok=True)

    jobs = []  # a list of the 'Job' class
    norm_contam_pcts = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for norm_pct in norm_contam_pcts:
        tumor_pct = 100 - norm_pct
        norm_ratio = norm_pct / 100
        tumor_ratio = tumor_pct / 100

        # in-loop path settings
        tag = 'n%st%s' % (norm_pct, tumor_pct)
        result_bam_path = result_bam_path_format % tag
        tumor_temp_bam_path = '%s/%s' % (temp_dir, os.path.basename(tumor_bam_path))
        tumor_temp_bam_path = tumor_temp_bam_path.replace('.bam', '.%s%%.bam' % tumor_pct)
        norm_temp_bam_path = '%s/%s' % (temp_dir, os.path.basename(norm_bam_path))
        norm_temp_bam_path = norm_temp_bam_path.replace('.bam', '.%s%%.bam' % norm_pct)

        cmd = ''
        # extracting some reads from the tumor bam file
        cmd += '%s view -b -s %.1f %s > %s;' % (samtools, (SEED + tumor_ratio), tumor_bam_path, tumor_temp_bam_path)
        # extracting some reads from the normal bam file
        cmd += '%s view -b -s %.1f %s > %s;' % (samtools, (SEED + norm_ratio), norm_bam_path, norm_temp_bam_path)
        # merging
        cmd += '%s -jar %s MergeSamFiles I=%s I=%s O=%s;' % \
               (java, picard, tumor_temp_bam_path, norm_temp_bam_path, result_bam_path)
        # indexing
        cmd += 'samtools index %s;' % result_bam_path

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
