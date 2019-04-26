#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Calculate MSE from the results of tumor purity estimation

* Prerequisite
    Run purity_estimator.py
"""
# param settings
cell_line = 'HCC1143'
depth = '30x'
kde_bandwidths = ['rule-of-thumb', 0.02, 0.03, 0.04, 0.05]

# path settings
project_dir = '/extdata4/baeklab/minwoo/projects/deep-purity'
purity_est_result_dir = f'{project_dir}/results/heuristic/est-tumor-purity'
outfile_path = f'{purity_est_result_dir}/mse_{cell_line}_{depth}.txt'

with open(outfile_path, 'w') as outfile:
    for kde_bandwidth in kde_bandwidths:
        if kde_bandwidth != 'rule-of-thumb':
            kde_bandwidth = round(kde_bandwidth, 2)

        purity_est_path = f'{purity_est_result_dir}/purity_{cell_line}_{depth}_{kde_bandwidth}.txt'
        se_sum = 0.0
        purity_cnt = 0

        with open(purity_est_path, 'r') as infile:
            for line in infile:
                cols = line.strip().split('\t')
                tumor_purity = float(cols[0])

                if tumor_purity == 0.975 or tumor_purity == 0.925:
                    continue

                est_purity = float(cols[1])
                se_sum += (tumor_purity - est_purity) ** 2
                purity_cnt += 1

        mse = round(se_sum / purity_cnt, 4)
        if kde_bandwidth == 'rule-of-thumb':
            print(f'{kde_bandwidth}', mse, sep='\t', file=outfile)
        else:
            print(f'{kde_bandwidth:.2f}', mse, sep='\t', file=outfile)

