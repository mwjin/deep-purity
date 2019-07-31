#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Execute Mutect (Cibulskis et al., 2013, Nat. Biotech.) to call the variants from our normal contaminated bam files.

* Prerequisite
    1. Run 01_in_silico_contam.py
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
    prev_job_prefix = 'Minu.In.Silico.Normal.Contam'
    job_name_prefix = 'Minu.Mutect.Variant.Call'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cell_line = 'HCC1143'
    depth = '30x'

    # path settings
    # input
    in_bam_dir = f'/extdata4/baeklab/minwoo/data/TCGA-HCC-DEPTH-NORM-MIX/{cell_line}/{depth}'
    in_bam_path_format = f'{in_bam_dir}/{cell_line}.%s.{depth}.bam'
    norm_bam_path = f'/extdata4/baeklab/minwoo/data/TCGA-HCC-DEPTH-NORM/' \
                    f'{cell_line}/{cell_line}.NORMAL.{depth}.bam'  # ctrl

    # output
    out_dir = f'{PROJECT_DIR}/results/mutect-output-depth-norm/{cell_line}/{depth}'
    out_vcf_path_format = f'{out_dir}/{cell_line}.%s.{depth}.vcf'
    out_mto_path_format = f'{out_dir}/{cell_line}.%s.{depth}.mto'
    os.makedirs(out_dir, exist_ok=True)

    ref_genome_dict = {
        'HCC1143': '/extdata6/Minwoo/data/ref-genome/hg19/Homo_sapiens_assembly19.fasta',
        'HCC1954': '/extdata6/Minwoo/data/ref-genome/hg19/Homo_sapiens_assembly19.fasta',
        'HCC1187': '/extdata6/Beomman/raw-data/ref/Homo_sapiens/NCBI/GRCh38Decoy/Sequence/WholeGenomeFasta/genome.fa',
        'HCC2218': '/extdata6/Beomman/raw-data/ref/Homo_sapiens/NCBI/GRCh38Decoy/Sequence/WholeGenomeFasta/genome.fa',
    }

    dbsnp_dict = {
        'HCC1143': '/extdata6/Beomman/raw-data/dbsnp/dbsnp150/compressed/hg19/common_all_20170710.hg19.vcf.gz',
        'HCC1954': '/extdata6/Beomman/raw-data/dbsnp/dbsnp150/compressed/hg19/common_all_20170710.hg19.vcf.gz',
        'HCC1187': '/extdata6/Beomman/raw-data/dbsnp/dbsnp150/compressed/hg38/common_all_20170710.hg38.vcf.gz',
        'HCC2218': '/extdata6/Beomman/raw-data/dbsnp/dbsnp150/compressed/hg38/common_all_20170710.hg38.vcf.gz',
    }

    # constant paths for mutect
    java = '/usr/java/latest/bin/java'  # must be JDK1.7
    temp_dir = f'{out_dir}/log'  # error log files will be stored.
    mutect = '/extdata6/Beomman/bins/mutect/mutect-1.1.7.jar'
    ref_genome = ref_genome_dict[cell_line]
    dbsnp_path = dbsnp_dict[cell_line]

    jobs = []  # a list of the 'Job' class
    norm_contam_pcts = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for norm_pct in norm_contam_pcts:
        tumor_pct = 100 - norm_pct
        tag = f'n{int(norm_pct)}t{int(tumor_pct)}'

        # in-loop path settings
        in_bam_path = in_bam_path_format % tag
        out_vcf_path = out_vcf_path_format % tag
        out_mto_path = out_mto_path_format % tag

        cmd = f'{java} -Xmx2g -Djava.io.tmpdir={temp_dir} -jar {mutect} -rf BadCigar --analysis_type MuTect ' \
              f'--reference_sequence {ref_genome} --dbsnp {dbsnp_path} ' \
              f'--input_file:normal {norm_bam_path} --input_file:tumor {in_bam_path} ' \
              f'--tumor_lod 6.3 --initial_tumor_lod 4.0 --fraction_contamination 0.0 ' \
              f'--vcf {out_vcf_path} -o {out_mto_path} --enable_extended_output'

        if is_test:
            print(cmd)
        else:
            prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{tag}'
            one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{tag}'
            one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
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
