#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
"""
Segment like CHAT using variants heterozygous in normal from cell lines to make test data
"""
from lab.job import Job, qsub_sge
from lab.utils import time_stamp
from settings import PROJECT_DIR

import os
import sys


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    queue = '24_730.q'
    is_test = True
    job_name_prefix = 'Minu.DeepPurity.CHAT.Segments'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # path settings
    var_tsv_dir = f'{PROJECT_DIR}/data/variants-tsv/original'
    seg_tsv_dir = f'{PROJECT_DIR}/data/segments-tsv/original'
    seg_script = f'{PROJECT_DIR}/src/segmentation.R'

    # path settings
    segment_size = 1000
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'
    norm_contams = list(range(5, 100, 5))

    jobs = []  # a list of the 'Job' class

    for cell_line in cell_lines:
        # in-loop path settings
        in_var_tsv_dir = f'{var_tsv_dir}/{cell_line}/{depth}'
        out_seg_tsv_dir = f'{seg_tsv_dir}/{cell_line}/{depth}'
        os.makedirs(out_seg_tsv_dir, exist_ok=True)

        test_data_paths = []

        for norm_contam in norm_contams:
            tumor_purity = 100 - norm_contam
            purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

            in_tsv_path = f'{in_var_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
            out_tsv_path = f'{out_seg_tsv_dir}/{cell_line}.{purity_tag}.{depth}.tsv'

            test_data_paths.append(out_tsv_path)
            cmd = f'{seg_script} {out_tsv_path} {in_tsv_path} {segment_size};'

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
