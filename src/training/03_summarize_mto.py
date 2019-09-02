#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
From MTO (MuTect output) file, extract and store essential information of the passed variants as TSV file.
This is for reducing memory overhead.

* Prerequisite
    1. Run 02_run_mutect.py
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp

import os
import sys

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.Mutect.Variant.Call'
    job_name_prefix = 'Minu.Summarize.Mutect.Output'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depths = ['30x']

    # path settings
    summary_script = f'{PROJECT_DIR}/src/summarize_mto.py'
    norm_contams = list(range(5, 100, 5))
    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # path settings
            mto_dir = f'{PROJECT_DIR}/data/mutect-output-depth-norm/{cell_line}/{depth}'
            mto_summary_dir = f'{PROJECT_DIR}/data/variants-tsv/original/{cell_line}/{depth}'
            os.makedirs(mto_summary_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                mto_path = f'{mto_dir}/{cell_line}.{purity_tag}.{depth}.mto'
                output_path = f'{mto_summary_dir}/{cell_line}.{purity_tag}.{depth}.tsv'

                cmd = f'{summary_script} {output_path} {mto_path}'

                if is_test:
                    print(cmd)
                else:
                    prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
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
            sys.exit(f'[ERROR]: The function \"{function_name}\" is unavailable.')
