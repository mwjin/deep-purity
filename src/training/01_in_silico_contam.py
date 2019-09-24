#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
This script is used to mix normal reads (a%) and tumor reads ((1 - a)%) then make virtual tumor data in silico.
(a: normal contamination ratio)
"""
from random import randint
from lab.job import Job, qsub_sge
from lab.utils import time_stamp
from settings import PROJECT_DIR

import os
import sys

# constants
SEED = randint(0, 1000000)


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
    benchmark_list = ['tcga-benchmark4', 'giab', 'platinum']
    cell_line_dict = {
        'tcga-benchmark4': ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218'],
        'giab': ['HG002-HG003', 'HG002-HG004'],
        'platinum': ['NA12877-NA12889', 'NA12877-NA12890', 'NA12878-NA12891', 'NA12878-NA12892']
    }
    depth_dict = {
        'tcga-benchmark4': ['30x'],
        'giab': ['30x', '100x', '150x'],
        'platinum': ['30x'],
    }

    benchmark = benchmark_list[1]
    cell_lines = cell_line_dict[benchmark]
    depths = depth_dict[benchmark]

    # path settings
    bam_dir_dict = {
        'tcga-benchmark4': '/extdata4/baeklab/minwoo/data/TCGA-HCC-DEPTH-NORM',
        'giab': '/extdata4/baeklab/minwoo/data/giab/illumina-paired-wes',
        'platinum': '/extdata4/baeklab/minwoo/data/platinum',
    }

    # script path settings
    samtools = '/home/lab/anaconda3/envs/Sukjun/bin/samtools'
    java = '/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/java'
    picard = '/extdata6/Beomman/bins/picard/build/libs/picard.jar'

    bam_dir = bam_dir_dict[benchmark]
    jobs = []  # a list of the 'Job' class
    norm_contams = list(range(5, 100, 5))

    for cell_line in cell_lines:
        for depth in depths:
            # in-loop path settings
            if benchmark == 'giab':
                bam_path_table = {
                    'HG002': f'{bam_dir}/{depth}/HG002_NA24835_son.bam',
                    'HG003': f'{bam_dir}/{depth}/HG003_NA24149_father.bam',
                    'HG004': f'{bam_dir}/{depth}/HG004_NA24143_mother.bam',
                }
                tumor_id, norm_id = cell_line.split('-')
                tumor_bam_path = bam_path_table[tumor_id]
                norm_bam_path = bam_path_table[norm_id]
                out_bam_dir = f'/extdata4/baeklab/minwoo/data/giab-mix/{cell_line}/{depth}'
            elif benchmark == 'platinum':
                bam_path_table = {
                    'NA12877': f'{bam_dir}/{depth}/NA12877.{depth}.bam',
                    'NA12878': f'{bam_dir}/{depth}/NA12878.{depth}.bam',
                    'NA12889': f'{bam_dir}/{depth}/NA12889.{depth}.bam',
                    'NA12890': f'{bam_dir}/{depth}/NA12890.{depth}.bam',
                    'NA12891': f'{bam_dir}/{depth}/NA12891.{depth}.bam',
                    'NA12892': f'{bam_dir}/{depth}/NA12892.{depth}.bam',
                }
                tumor_id, norm_id = cell_line.split('-')
                tumor_bam_path = bam_path_table[tumor_id]
                norm_bam_path = bam_path_table[norm_id]
                out_bam_dir = f'/extdata4/baeklab/minwoo/data/platinum-mix/{cell_line}/{depth}'
            else:  # tcga-benchmark4
                tumor_bam_path = f'{bam_dir}/{cell_line}/{cell_line}.TUMOR.{depth}.bam'
                norm_bam_path = f'{bam_dir}/{cell_line}/{cell_line}.NORMAL.{depth}.bam'
                out_bam_dir = f'{bam_dir}-MIX/{cell_line}/{depth}'

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
