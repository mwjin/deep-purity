#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Test bimodality of VAF histograms using Hartigan's dip / Silverman test
"""
import os
import sys

from lab.job import Job, qsub_sge
from lab.utils import time_stamp

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    # job scheduler settings
    queue = '24_730.q'
    is_test = True

    job_name_prefix = 'Minu.VAF.Hist.Bimodal.Test'
    prev_job_prefix = 'Minu.VAF.KS-Test.and.Filtering'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1187', 'HCC1954', 'HCC2218']
    depths = ['10x', '20x', '30x', '40x', '50x']

    # path setting
    rscript_bin = '/extdata6/Doyeon/anaconda3/envs/r-env/bin/Rscript'
    rscript_path = f'{PROJECT_DIR}/heuristic/dip_silverman_test.R'
    result_dir = f'{PROJECT_DIR}/results/heuristic'
    os.makedirs(result_dir, exist_ok=True)

    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        # in-loop path setting
        test_out_dir = f'{result_dir}/bimodal-test/{cell_line}'
        os.makedirs(test_out_dir, exist_ok=True)

        for depth in depths:
            # in-loop path setting
            var_tsv_dir = f'{result_dir}/ks-test/{cell_line}/{depth}'  # input
            cmd = f'{rscript_bin} {rscript_path} {test_out_dir} {var_tsv_dir} {cell_line} {depth};'

            if is_test:
                print(cmd)
            else:
                one_job_name = f'{job_name_prefix}.{cell_line}.{depth}'
                prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.*'
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
            sys.exit(f'[ERROR] There is no \'{function_name}\' function.')
