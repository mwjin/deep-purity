#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
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
    job_name_prefix = 'Minu.In.Silico.Normal.Contam'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depths = ['30x']

    # script path settings
    samtools = '/home/lab/anaconda3/envs/Sukjun/bin/samtools'
    java = '/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/java'
    picard = '/extdata6/Beomman/bins/picard/build/libs/picard.jar'

    # path settings
    bam_dir = '/extdata4/baeklab/minwoo/data/TCGA-HCC-DEPTH-NORM'
    jobs = []  # a list of the 'Job' class
    norm_contams = list(range(5, 100, 5))

    for cell_line in cell_lines:
        for depth in depths:
            # in-loop path settings
            tumor_bam_path = f'{bam_dir}/{cell_line}/{cell_line}.TUMOR.{depth}.bam'
            norm_bam_path = f'{bam_dir}/{cell_line}/{cell_line}.NORMAL.{depth}.bam'
            out_bam_dir = f'/extdata4/baeklab/minwoo/data/TCGA-HCC-DEPTH-NORM-MIX/{cell_line}/{depth}'
            temp_dir = f'{out_bam_dir}/temp'
            os.makedirs(temp_dir, exist_ok=True)

            for contam in norm_contams:
                purity = 100 - contam
                norm_ratio = contam / 100
                tumor_ratio = purity / 100

                # in-loop path settings
                purity_tag = f'n{int(contam)}t{int(purity)}'
                out_bam_path = f'{out_bam_dir}/{cell_line}.{purity_tag}.{depth}.bam'

                tumor_temp_bam_path = f'{temp_dir}/{os.path.basename(tumor_bam_path)}'
                tumor_temp_bam_path = tumor_temp_bam_path.replace('.bam', '.%d%%.bam' % purity)
                norm_temp_bam_path = f'{temp_dir}/{os.path.basename(norm_bam_path)}'
                norm_temp_bam_path = norm_temp_bam_path.replace('.bam', '.%d%%.bam' % contam)

                cmd = f'{samtools} view -b -s {(SEED + tumor_ratio):.3f} {tumor_bam_path} > {tumor_temp_bam_path};' \
                      f'{samtools} view -b -s {(SEED + norm_ratio):.3f} {norm_bam_path} > {norm_temp_bam_path};' \
                      f'{java} -jar {picard} MergeSamFiles I={tumor_temp_bam_path} I={norm_temp_bam_path} ' \
                      f'O={out_bam_path};' \
                      f'{samtools} index {out_bam_path};'

                if is_test:
                    print(cmd)
                else:
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job = Job(one_job_name, cmd)
                    jobs.append(one_job)

            # for removing the temporary directory
            temp_rm_cmd = f'rm -rf {temp_dir};'

            if is_test:
                print(temp_rm_cmd)
            else:
                rm_job_name = f'{job_name_prefix}.{cell_line}.{depth}.Remove.Temp.Dir'
                prev_job_name = f'{job_name_prefix}.{cell_line}.{depth}.*'
                rm_job = Job(rm_job_name, temp_rm_cmd, hold_jid=prev_job_name)
                jobs.append(rm_job)

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
