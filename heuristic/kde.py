#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Gaussian kernel density estimation of VAF histograms
"""
import numpy
import matplotlib.pyplot as plt
import os
import sys

from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity
from lab.utils import eprint, time_stamp
from lab.job import Job, qsub_sge

# constants
PROJECT_DIR = '/extdata4/baeklab/minwoo/projects/deep-purity'


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    script = os.path.abspath(__file__)
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.Purity.Heuristic.VAF.KS-Test.and.Filtering'
    job_name_prefix = 'Minu.Purity.Heuristic.VAF.Hist.KDE'
    log_dir = f'{PROJECT_DIR}/log/heuristic/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent
    kde_bandwidth = 0.05
    top_var_cnt = 3200

    # path settings
    var_tsv_dir = f'{PROJECT_DIR}/results/heuristic/filt-variants-top-lodt/{top_var_cnt}'
    kde_result_dir = f'{PROJECT_DIR}/results/heuristic/kde-normal-top-lodt/{top_var_cnt}'

    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # in-loop path settings
            ks_test_out_dir = f'{var_tsv_dir}/{cell_line}/{depth}'  # input
            kde_out_dir = f'{kde_result_dir}/{cell_line}/{depth}'  # output
            os.makedirs(kde_out_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                # path2: a path of a plot after finding local extreama
                var_tsv_path = f'{ks_test_out_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                kde_result_path = f'{kde_out_dir}/kde_result_{cell_line}_{purity_tag}_{depth}.txt'
                kde_plot_path = f'{kde_out_dir}/plot_{cell_line}_{purity_tag}_{depth}.png'
                kde_plot_title = f'Gaussian_KDE_{cell_line}_{depth}_{purity_tag}'
                local_extrema_txt_path = f'{kde_out_dir}/local_extreme_vaf_{purity_tag}_{depth}.txt'

                if os.path.isfile(var_tsv_path):
                    cmd = f'{script} vaf_hist_kde {kde_result_path} {var_tsv_path} {kde_bandwidth};'
                    cmd += f'{script} find_local_extrema {local_extrema_txt_path} {kde_result_path};'
                    cmd += f'{script} draw_plot {kde_plot_path} {var_tsv_path} {kde_result_path} ' \
                           f'{local_extrema_txt_path} {kde_plot_title} {kde_bandwidth};'

                    if is_test:
                        print(cmd)
                    else:
                        one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                        prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                        one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                        jobs.append(one_job)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def vaf_hist_kde(kde_result_path, var_tsv_path, kde_bandwidth):
    """
    The function uses kernel density estimation for VAF histogram.
    The kernel density estimation smooths the shape of VAF histogram.

    :param kde_result_path: a path of a KDE result
    :param var_tsv_path: a path of a TSV file for variants
    :param kde_bandwidth: a bandwidth for KDE
    """
    eprint('[LOG] Construct a VAF histrogram')
    vaf_list = []

    with open(var_tsv_path, 'r') as var_tsv_file:
        var_tsv_file.readline()  # remove a header

        for line in var_tsv_file:
            cols = line.strip().split('\t')
            vaf_list.append(float(cols[5]))

    vaf_list = numpy.array(vaf_list)
    vaf_hist = {}  # key: VAF bins, value: count; dictionary for histogram

    if kde_bandwidth == 'rule-of-thumb':
        kde_bandwidth = _get_kde_bandwidth(vaf_list)
    else:
        kde_bandwidth = eval(kde_bandwidth)

    # initialization
    for i in range(101):
        vaf_hist[round(0.01 * i, 2)] = 0

    for vaf in vaf_list:
        vaf_hist[round(vaf, 2)] += 1

    eprint('[LOG] Gaussian kernel density estimation')
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(vaf_list[:, numpy.newaxis])
    xs = numpy.linspace(0, 1, len(vaf_list))[:, numpy.newaxis]
    log_kde_dens = kde.score_samples(xs)

    with open(kde_result_path, 'w') as kde_result_file:
        for x, y in zip(xs[:, 0], numpy.exp(log_kde_dens)):
            kde_result_file.write(f'{x}\t{y}\n')


def find_local_extrema(local_extrema_txt_path, kde_result_path):
    """
    The function finds local minima and local maxima from smoothed histogram.
    By finding local extrema, we can divide homozygous variants and heterozygous variants.
    """
    eprint(f'[LOG] Find local minima and maxima of KDE curve')
    list_x = []
    list_kde = []

    with open(kde_result_path, 'r') as kde_file:
        for line in kde_file:
            cols = line.strip('\n').split('\t')
            list_x.append(float(cols[0]))
            list_kde.append(float(cols[1]))

    list_kde = numpy.array(list_kde)
    list_kde = list_kde / list_kde.max()  # normalization

    local_maxima_indices = argrelextrema(list_kde, numpy.greater)[0]  # find local maximum
    local_minima_indices = argrelextrema(list_kde, numpy.less)[0]  # find local minimum

    local_maxima_indices = _local_extrema_filter(local_maxima_indices, list_x, list_kde, numpy.greater)
    local_minima_indices = _local_extrema_filter(local_minima_indices, list_x, list_kde, numpy.less)

    eprint(f'[LOG] Save local minima and maxima')
    vaf_to_label = {}  # key: a VAF, value: lmax (local maximum) or lmin (local minimum)

    for lmax_idx in local_maxima_indices:
        lmax_vaf = list_x[lmax_idx]
        vaf_to_label[lmax_vaf] = 'lmax'

    for lmin_idx in local_minima_indices:
        lmin_vaf = list_x[lmin_idx]
        vaf_to_label[lmin_vaf] = 'lmin'

    with open(local_extrema_txt_path, 'w') as local_extrema_txt_file:
        for vaf in sorted(vaf_to_label.keys()):
            print(vaf, vaf_to_label[vaf], sep='\t', file=local_extrema_txt_file)


def draw_plot(out_plot_path, var_tsv_path, kde_result_path, local_extrema_path, kde_plot_title, kde_bandwidth):
    """
    Draw a plot of the VAF histogram, KDE estimation result, and its local extrema
    """
    vaf_list = []

    eprint('[LOG] Construct a VAF histrogram for plotting')
    with open(var_tsv_path, 'r') as var_tsv_file:
        var_tsv_file.readline()  # remove a header

        for line in var_tsv_file:
            cols = line.strip().split('\t')
            vaf_list.append(float(cols[5]))

    vaf_list = numpy.array(vaf_list)
    vaf_hist = {}  # key: VAF bins, value: count; dictionary for histogram

    if kde_bandwidth == 'rule-of-thumb':
        kde_bandwidth = _get_kde_bandwidth(vaf_list)
    else:
        kde_bandwidth = eval(kde_bandwidth)

    # initialization
    for i in range(101):
        vaf_hist[round(0.01 * i, 2)] = 0

    for vaf in vaf_list:
        vaf_hist[round(vaf, 2)] += 1

    eprint(f'[LOG] Read KDE results')
    list_x = []
    list_kde = []

    with open(kde_result_path, 'r') as kde_file:
        for line in kde_file:
            cols = line.strip('\n').split('\t')
            list_x.append(float(cols[0]))
            list_kde.append(float(cols[1]))

    list_kde = numpy.array(list_kde)
    list_kde = list_kde / list_kde.max()  # normalization

    eprint(f'[LOG] Read local extrema')
    vaf_to_label = {}  # key: a VAF, value: lmax (local maximum) or lmin (local minimum)

    with open(local_extrema_path, 'r') as lextrema_file:
        for line in lextrema_file:
            cols = line.strip().split('\t')
            vaf_to_label[float(cols[0])] = cols[1]

    # get original tumor purity
    purity_tag = kde_plot_title.split('_')[-1]
    purity_pct = int(purity_tag.split('t')[1])
    tumor_purity = purity_pct / 100

    eprint(f'[LOG] Draw a plot containing all information')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('VAF')
    ax1.set_ylabel('Density')
    plt.title(kde_plot_title)

    # plot KDE results
    ax1.plot(list_x, list_kde, color='#8a8a8a', label=f'bandwidth: {kde_bandwidth:.2f}')

    # plot local extrema
    for lextrema_vaf in vaf_to_label:
        label = vaf_to_label[lextrema_vaf]

        if label == 'lmax':
            ax1.axvline(lextrema_vaf, color='red', linestyle='--')
        else:
            ax1.axvline(lextrema_vaf, color='blue', linestyle='--')

    # plot original tumor purity
    ax1.axvline(tumor_purity / 2, color='black', linestyle='-.')  # heterozygous
    ax1.axvline(tumor_purity, color='black', linestyle='-.')  # homozygous

    ax1.legend(loc='upper left')

    # plot the VAF histogram
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency')
    var_cnt = len(vaf_list)
    ax2.plot([round(x, 2) for x in sorted(vaf_hist.keys())],
             [vaf_hist[x] / var_cnt for x in sorted(vaf_hist.keys())],
             'g--', label=f'histogram (N: {var_cnt})')

    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 0.15)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(out_plot_path)
    plt.close()


def _get_kde_bandwidth(values):
    """
    Calculate an appropriate bandwidth using a rule-of-thumb bandwidth estimator (Silverman, 1986)
    """
    val_cnt = len(values)
    val_std = float(numpy.std(values))
    bandwidth = ((4 * val_std ** 5) / (3 * val_cnt)) ** (1 / 5)

    return bandwidth


def _local_extrema_filter(lextrema_indices, xs, ys, comparator):
    """
    Get only major peaks from the local extrema
    This function see trends of left and right values of each peak.
    It checks whether absolute values of differences
    between the local extreme value and a left or right value globally increases or not.
    """
    # params
    x_cnt = len(xs)
    max_x_interval = 0.1
    x_interval = xs[1] - xs[0]

    num_check = 20
    max_idx_interval = int(max_x_interval // x_interval)
    idx_interval = max_idx_interval // num_check

    # if interval is too small, reduce # of checks
    while idx_interval == 0:
        num_check //= 2
        idx_interval = max_idx_interval // num_check

    penalty_cutoff = num_check // 2
    print(x_interval, max_idx_interval, idx_interval, num_check)

    filtered_lextrema_indices = []

    for lextrema_idx in lextrema_indices:
        print(lextrema_idx, xs[lextrema_idx])
        lextrema = ys[lextrema_idx]

        left_diffs = []
        right_diffs = []

        left_idx = lextrema_idx
        right_idx = lextrema_idx

        # update slopes
        for i in range(num_check):
            left_idx = lextrema_idx - idx_interval * (i + 1)

            if left_idx < 0:
                left_idx = 0

            left_val = ys[left_idx]
            left_diff = lextrema - left_val
            left_diffs.append(left_diff)

            right_idx = lextrema_idx + idx_interval * (i + 1)

            if right_idx >= x_cnt:
                right_idx = x_cnt - 1

            right_val = ys[right_idx]
            right_diff = lextrema - right_val
            right_diffs.append(right_diff)

        print(left_diffs, right_diffs)

        # check whether the slopes globally increase or not
        left_penalty = 0  # +1 if trend is reverse
        right_penalty = 0

        for i in range(1, num_check):
            if comparator(left_diffs[i - 1], left_diffs[i]):
                left_penalty += 1

            if comparator(right_diffs[i - 1], right_diffs[i]):
                right_penalty += 1

        left_slope = left_diffs[-1] / (xs[lextrema_idx] - xs[left_idx])
        right_slope = right_diffs[-1] / (xs[lextrema_idx] - xs[right_idx])
        print(left_slope, right_slope)

        if left_penalty < penalty_cutoff and right_penalty < penalty_cutoff:
            if (left_slope * right_slope < 0) \
                    and (round(abs(left_slope), 1) >= 0.2 and round(abs(right_slope), 1) >= 0.2):
                filtered_lextrema_indices.append(lextrema_idx)

        print(left_penalty, right_penalty)

    return numpy.array(filtered_lextrema_indices)


def vaf_hist_kde_old(out_dir, ks_result_dir, diptest_result_path, silverman_result_path):
    """
    Deprecated function
    """
    """
    we uses both of dip test and silverman test for bimodality test.
    silverman test is used for p value and dip test for VAF threshold.
    if silverman test's p value < 0.01, we decide the variants have bimodal distribution.

    """
    cell_line = ''
    depth = ''
    purity_tags = []

    eprint('[LOG] Parse previous bimodal test results')
    purity_tag_to_vaf = {}  # key: 'n10t90', value: 0.5 (VAF cutoff)

    with open(diptest_result_path, 'r') as diptest_file:
        diptest_file.readline()

        for line in diptest_file:
            cols = line.strip('\n').split('\t')
            cell_line = cols[0]
            depth = cols[1]
            purity_tag = cols[2]
            purity_tags.append(purity_tag)
            purity_tag_to_vaf[purity_tag] = float(cols[4])

    purity_tag_to_silverman = {}  # key: 'n10t90', value: 0.001 (p value)

    with open(silverman_result_path, 'r') as silverman_result_file:
        silverman_result_file.readline()

        for line in silverman_result_file:
            cols = line.strip('\n').split('\t')
            purity_tag_to_silverman[cols[2]] = float(cols[3])

    # KDE
    eprint(f'[LOG] Current cell line: {cell_line}')
    eprint(f'[LOG] Current depth: {depth}')

    for purity_tag in purity_tags:
        eprint(f'[LOG] Current purity: {purity_tag}')
        # in-loop path setting
        var_tsv_path = f'{ks_result_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
        ks_result_path = f'{ks_result_dir}/ks_test_result_{purity_tag}.txt'
        kde_plot_path = f'{out_dir}/kde_curve_{purity_tag}.png'
        kde_result_path = f'{out_dir}/kde_result_{purity_tag}.txt'

        vaf_list = []

        with open(var_tsv_path, 'r') as var_tsv_file:
            var_tsv_file.readline()  # remove a header

            for line in var_tsv_file:
                cols = line.strip().split('\t')
                vaf_list.append(float(cols[5]))

        # construct VAF histogram dictionary
        eprint(f'[LOG] --- Construct VAF histogram')
        vaf_hist = dict()  # key: VAF bins, value: count;
        bins = [round(0.01 * x, 2) for x in range(101)]  # element: [VAF1, VAF2)

        for idx in range(len(bins) - 1):
            vaf_hist[(bins[idx], bins[idx + 1])] = 0  # initialize by 0

        for vaf in vaf_list:
            check = False  # check if data is count

            for key in sorted(vaf_hist.keys(), key=lambda x: x[0]):
                if key[0] <= vaf < key[1]:  # binning for histogram
                    vaf_hist[key] += 1
                    check = True
                    break

            if not check:  # if data is not counted, the data is 1.
                if vaf == 1.00:
                    vaf_hist[(0.99, 1.00)] += 1
                else:
                    sys.exit(f'[ERROR] --- Invalid VAF: {vaf}')

        with open(ks_result_path, 'r') as ks_result_file:
            ks_result_file.readline()  # remove a header
            stats = ks_result_file.readline().strip('\n').split('\t')

        # two independent plot(kde smooth hist and origin hist) are drawn at single figure.
        eprint(f'[LOG] --- Draw histogram plot for both KDE and original')
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('VAF')
        ax1.set_ylabel('Density')  # kde density
        plt.title('gaussian kde %s %s %s' % (cell_line, depth, purity_tag))

        kde_bandwidth = 0.01  # kde parameter; the histogram is more smooth at low bandwidth.
        color = 'red'
        kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(vaf_list[:, numpy.newaxis])
        x_plot = numpy.linspace(0, 1, len(vaf_list))[:, numpy.newaxis]
        log_dens = kde.score_samples(x_plot)
        ax1.plot(x_plot[:, 0], numpy.exp(log_dens), color=color, label='bandwidth: %s' % kde_bandwidth, alpha=0.5)
        ax1.legend(loc='upper right')

        ax2 = ax1.twinx()  # origin histogram
        ax2.set_ylabel('Frequency')
        total_count_high_lodt = sum(vaf_hist.values())
        ax2.plot([round((x[0] + x[1]) / 2, 2) for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
                 [vaf_hist[x] / total_count_high_lodt for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
                 'g--', label='histogram(high LODt)\nN: %s cutoff: %s diff: %s' %
                              (total_count_high_lodt, stats[3], round(float(stats[4]), 4)))

        if purity_tag_to_silverman['%s_%s_%s' % (cell_line, depth, purity_tag)] < 0.01:
            plt.axvline(x=purity_tag_to_vaf['%s_%s_%s' % (cell_line, depth, purity_tag)])

        ax2.tick_params(axis='y')
        ax2.legend(loc='center right')

        fig.tight_layout()
        plt.savefig(kde_plot_path)
        plt.close()

        eprint(f'[LOG] --- Save KDE result')
        with open(kde_result_path, 'w') as kde_out_file:
            for x, y in zip(x_plot[:, 0], numpy.exp(log_dens)):
                kde_out_file.write('%s\t%s\n' % (x, y))


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
