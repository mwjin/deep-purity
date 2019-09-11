#!/home/sonic/baeklab/Hyeonseong/anaconda3/envs/minwoo/bin/python
"""
Evaluate the results of prediction by calculating the mean squared error (MSE) for each cell line

* Prerequisite
    1. Run 04_test_model.py
"""


def main():
    # param settings
    cell_lines = ['HCC1143', 'HCC1954', 'HCC1187', 'HCC2218']
    depth = '30x'
    model_ver = '190911-cnn1'
    tumor_purities = list(range(5, 100, 5))

    # path settings
    project_dir = '/extdata1/baeklab/minwoo/projects/deep-purity'
    pred_result_dir = f'{project_dir}/results/prediction/{model_ver}'  # input
    mse_result_path = f'{pred_result_dir}/mse_{depth}.txt'  # output

    with open(mse_result_path, 'w') as outfile:
        for cell_line in cell_lines:
            pred_result_path = f'{pred_result_dir}/result_{cell_line}_{depth}.txt'
            pred_result_dict = {}  # key: actual purity, value: predicted purity

            with open(pred_result_path, 'r') as pred_result_file:
                for line in pred_result_file:
                    cols = line.strip().split('\t')
                    pred_result_dict[float(cols[1])] = float(cols[2])

            mse = 0

            for tumor_purity in tumor_purities:
                actual_purity = round(tumor_purity / 100, 2)
                predicted_purity = pred_result_dict[actual_purity]
                mse += (actual_purity - predicted_purity) ** 2

            mse /= len(tumor_purities)

            print(cell_line, round(mse, 4), sep='\t', file=outfile)


if __name__ == '__main__':
    main()
