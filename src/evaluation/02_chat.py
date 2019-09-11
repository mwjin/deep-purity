#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Segment like CHAT using variants heterozygous in normal from cell lines to make test data
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
    job_name_prefix = 'Minu.DeepPurity.CHAT'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    seg_tsv_dir = f'{PROJECT_DIR}/data/segments-tsv/original'
    chat_tsv_dir = f'{PROJECT_DIR}/data/chat-tsv/original'
    chat_script = f'{PROJECT_DIR}/src/chat_mock.py'

    # path settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'
    norm_contams = list(range(5, 100, 5))

    jobs = []  # a list of the 'Job' class

    for cell_line in cell_lines:
        # in-loop path settings
        in_seg_tsv_dir = f'{seg_tsv_dir}/{cell_line}/{depth}'
        out_chat_tsv_dir = f'{chat_tsv_dir}/{cell_line}/{depth}'
        os.makedirs(out_chat_tsv_dir, exist_ok=True)

        test_data_paths = []

        for norm_contam in norm_contams:
            tumor_purity = 100 - norm_contam
            purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

            in_tsv_path = f'{in_seg_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
            out_tsv_path = f'{out_chat_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'

            test_data_paths.append(out_tsv_path)
            cmd = f'{chat_script} {out_tsv_path} {in_tsv_path};'

            if is_test:
                print(cmd)
            else:
                one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
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
            sys.exit(f'[ERROR]: The function \"{function_name}\" is unavailable.')
