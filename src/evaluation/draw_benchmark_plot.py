#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
import os
import sys
import numpy
import matplotlib.pyplot as plt


def make_benchmark_plot():
    """
    Make a plot for each tumor purity estimation tools for benchmarking
    """
    # params
    ratios = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # tumor ratio for each normal contam
    cell_line = sys.argv[1]
    depth = '30x'
    model_ver = '190523.5-2'  # for deep-purity
    top_lodt_num = 3100 if cell_line == 'HCC1143' else 3500  # for heuristic

    # key : actual tumor purity level
    # value : tumor purity estimation value from each tool (list type, absolute-chat-deeppurity order)
    purity_to_predictions = dict()
    
    # for absolute
    abs_summary_path = f'/extdata4/baeklab/minwoo/projects/absolute/results/absolute-depth-norm/' \
                       f'{cell_line}/{depth}/purity_summary.txt'

    with open(abs_summary_path, 'r') as abs_file:
        for line in abs_file:
            cols = line.strip('\n').split()
            true_ratio = round(float(cols[0]), 3)  # transform to tumor purity level
            pred_ratio = round(float(cols[1]), 3)  # transform to tumor purity estimation

            if true_ratio not in ratios:
                continue

            purity_to_predictions[true_ratio] = [pred_ratio]

    # for chat
    chat_summary_path = f'/extdata4/baeklab/minwoo/projects/chat/results/depth-norm/' \
                        f'{cell_line}/{depth}/purity_summary.txt'

    with open(chat_summary_path, 'r') as chat_result_file:
        for line in chat_result_file:
            cols = line.strip().split('\t')
            true_ratio = round(float(cols[0]), 3)
            pred_ratio = round(float(cols[1]), 3)

            if true_ratio not in ratios:
                continue

            purity_to_predictions[true_ratio].append(pred_ratio)

    # for deep-purity
    deep_purity_path = f'/extdata4/baeklab/minwoo/projects/deep-purity/results/prediction/' \
                       f'result_{cell_line}_{depth}.txt'
    deep_purity_outfile = open(deep_purity_path, 'r')

    for line in deep_purity_outfile:
        cols = line.strip('\n').split()
        true_ratio = round(float(cols[1]), 3)
        if true_ratio not in ratios:
            continue
        pred_ratio = round(float(cols[2]), 3)
        purity_to_predictions[true_ratio].append(pred_ratio)

    # for our heuristic model
    heuristic_path = f'/extdata4/baeklab/minwoo/projects/deep-purity/results/heuristic/est-tumor-purity/' \
                     f'top_lodt_purity_{cell_line}_30x_{top_lodt_num}.txt'
    heuristic_out_file = open(heuristic_path, 'r')

    for line in heuristic_out_file:
        cols = line.strip('\n').split()
        true_ratio = round(float(cols[0]), 3)
        if true_ratio not in ratios:
            continue
        pred_ratio = round(float(cols[1]), 3)
        purity_to_predictions[true_ratio].append(pred_ratio)

    heuristic_out_file.close()

    for purity in purity_to_predictions:
        print(purity, purity_to_predictions[purity])

    # for plotting
    out_dir = '/extdata4/baeklab/minwoo/projects/deep-purity/results/benchmark'
    png_outfile = f'{out_dir}/190809_{cell_line}_{depth}_benchmark.png'
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.title(f'Purity Estimation Benchmark {cell_line}', size=15)
    mse_abs = round(float(numpy.mean([(x - purity_to_predictions[x][0]) ** 2 for x in ratios])), 4)
    plt.plot(sorted(ratios), [purity_to_predictions[x][0] for x in sorted(ratios)],
             color='#f72020', marker='o', linestyle='dashed', markersize=8, label=f'Absolute(MSE:{mse_abs})')

    mse_chat = round(float(numpy.mean([(x - purity_to_predictions[x][1]) ** 2 for x in ratios])), 4)
    plt.plot(sorted(ratios), [purity_to_predictions[x][1] for x in sorted(ratios)],
             color='#2020f7', marker='o', linestyle='dashed', markersize=8, label=f'CHAT(MSE:{mse_chat})')

    mse_dp = round(float(numpy.mean([(x - purity_to_predictions[x][2]) ** 2 for x in ratios])), 4)
    plt.plot(sorted(ratios), [purity_to_predictions[x][2] for x in sorted(ratios)],
             color='#257c3c', marker='o', linestyle='dashed', markersize=8, label=f'DeepPurity(MSE:{mse_dp})')

    mse_heu = round(float(numpy.mean([(x - purity_to_predictions[x][3]) ** 2 for x in ratios])), 4)
    plt.plot(sorted(ratios), [purity_to_predictions[x][3] for x in sorted(ratios)],
             color='#e9e13c', marker='o', linestyle='dashed', markersize=8, label=f'Heuristic(MSE:{mse_heu})')

    plt.plot(sorted(ratios), sorted(ratios), 'ko-', markersize=8)  # for actual purities

    plt.ylim((0.0, 1.0))
    plt.ylabel('Predicted tumor purity', size=12)
    plt.xlim((0.0, 1.0))
    plt.xlabel('Actual tumor purity', size=12)
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(False)
    plt.xticks(sorted(ratios), sorted(ratios), size=12)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], size=12)
    plt.savefig(png_outfile)
    plt.close()
    print('hello')


if __name__ == '__main__':
    make_benchmark_plot()
