#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def make_benchmark_plot():
    """
    Make a plot for each tumor purity estimation tools for benchmarking
    """
    # params
    purities = np.round(np.arange(0.05, 1.00, 0.05), 3)
    cell_line = sys.argv[1]
    depth = '30x'
    dp_model_ver = '190906-cnn1-1'  # for deep-purity

    # key : actual tumor purity level
    # value : tumor purity estimation value from each tool (list type, absolute-chat-deeppurity order)
    pred_purity_dict = {
        'absolute': [],
        'chat': [],
        'deep-purity': [],
    }
    
    # for absolute
    abs_summary_path = f'/extdata4/baeklab/minwoo/projects/absolute/results/absolute-depth-norm/' \
                       f'{cell_line}/{depth}/purity_summary.txt'
    abs_true_to_pred = {}

    with open(abs_summary_path, 'r') as abs_file:
        for line in abs_file:
            cols = line.strip('\n').split()
            true_purity = round(float(cols[0]), 3)  # transform to tumor purity level
            pred_purity = round(float(cols[1]), 3)  # transform to tumor purity estimation
            abs_true_to_pred[true_purity] = pred_purity

    # for chat
    chat_summary_path = f'/extdata4/baeklab/minwoo/projects/chat/results/depth-norm/' \
                        f'{cell_line}/{depth}/purity_summary.txt'
    chat_true_to_pred = {}

    with open(chat_summary_path, 'r') as chat_result_file:
        for line in chat_result_file:
            cols = line.strip().split('\t')
            true_purity = round(float(cols[0]), 2)
            pred_purity = round(float(cols[1]), 2)
            chat_true_to_pred[true_purity] = pred_purity

    # for deep-purity
    dp_result_path = f'/extdata4/baeklab/minwoo/projects/deep-purity/results/prediction/{dp_model_ver}/' \
                     f'result_{cell_line}_{depth}.txt'
    dp_true_to_pred = {}

    with open(dp_result_path, 'r') as dp_result_file:
        for line in dp_result_file:
            cols = line.strip('\n').split()
            true_purity = round(float(cols[1]), 3)
            pred_purity = round(float(cols[2]), 3)
            dp_true_to_pred[true_purity] = pred_purity

    for purity in purities:
        pred_purity_dict['chat'].append(chat_true_to_pred[purity])
        pred_purity_dict['absolute'].append(abs_true_to_pred[purity])
        pred_purity_dict['deep-purity'].append(dp_true_to_pred[purity])

    # for plotting
    out_dir = '/extdata4/baeklab/minwoo/projects/deep-purity/results/benchmark'
    png_outfile = f'{out_dir}/{dp_model_ver}_{cell_line}_{depth}_benchmark.png'
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title(f'Purity Estimation Benchmark {cell_line}', size=15)
    mse_abs = round(float(np.mean((np.array(pred_purity_dict['absolute']) - purities) ** 2)), 4)
    plt.plot(purities, pred_purity_dict['absolute'],
             color='#f72020', marker='o', linestyle='dashed', markersize=8, label=f'Absolute(MSE:{mse_abs})')

    mse_chat = round(float(np.mean((np.array(pred_purity_dict['chat']) - purities) ** 2)), 4)
    plt.plot(purities, pred_purity_dict['chat'],
             color='#2020f7', marker='o', linestyle='dashed', markersize=8, label=f'CHAT(MSE:{mse_chat})')

    mse_dp = round(float(np.mean((np.array(pred_purity_dict['deep-purity']) - purities) ** 2)), 4)
    plt.plot(purities, pred_purity_dict['deep-purity'],
             color='#257c3c', marker='o', linestyle='dashed', markersize=8, label=f'DeepPurity(MSE:{mse_dp})')

    plt.plot(purities, purities, 'ko-', markersize=8)  # for actual purities

    plt.ylim((0.0, 1.0))
    plt.ylabel('Predicted tumor purity', size=12)
    plt.xlim((0.0, 1.0))
    plt.xlabel('Actual tumor purity', size=12)
    plt.legend(loc='lower right', fontsize='large')
    plt.grid(alpha=0.6, linestyle='--')
    plt.xticks(purities, purities, size=11)
    plt.yticks(purities, purities, size=11)
    plt.savefig(png_outfile)
    plt.close()


if __name__ == '__main__':
    make_benchmark_plot()
