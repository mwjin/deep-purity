#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
# TODO
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
KDE_BANDWIDTH = 0.05  # kde parameter; the histogram is more smooth at low bandwidth.


def main():
    """
    Bootstrap
    """
    # job scheduler settings
    script = os.path.abspath(__file__)
    queue = '24_730.q'
    is_test = True

    prev_job_prefix = 'Minu.VAF.KS-Test.and.Filtering'
    job_name_prefix = 'Minu.VAF.Hist.KDE'
    log_dir = f'{PROJECT_DIR}/log/{job_name_prefix}/{time_stamp()}'

    # param settings
    cells = ['HCC1143', 'HCC1954']
    depths = ['30x']
    norm_contams = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # unit: percent

    # path settings
    ks_test_result_dir = f'{PROJECT_DIR}/results/heuristic/ks-test'
    kde_result_dir = f'{PROJECT_DIR}/results/heuristic/kde/{KDE_BANDWIDTH}'

    jobs = []  # a list of the 'Job' class

    for cell_line in cells:
        for depth in depths:
            # in-loop path settings
            ks_test_out_dir = f'{ks_test_result_dir}/{cell_line}/{depth}'  # input
            kde_out_dir = f'{kde_result_dir}/{cell_line}/{depth}'  # output
            os.makedirs(kde_out_dir, exist_ok=True)

            for norm_contam in norm_contams:
                tumor_purity = 100 - norm_contam
                purity_tag = f'n{int(norm_contam)}t{int(tumor_purity)}'

                # in-loop path settings
                # path2: a path of a plot after finding local extreama
                var_tsv_path = f'{ks_test_out_dir}/{cell_line}.{purity_tag}.{depth}.tsv'
                kde_result_path = f'{kde_out_dir}/kde_result_{cell_line}_{purity_tag}_{depth}.txt'
                kde_plot_path = f'{kde_out_dir}/kde_curve_{cell_line}_{purity_tag}_{depth}.png'
                kde_plot_path2 = f'{kde_out_dir}/kde_curve_2_{cell_line}_{purity_tag}_{depth}.png'
                kde_plot_title = f'Gaussian_KDE_{cell_line}_{depth}_{purity_tag}'
                local_extrema_txt_path = f'{kde_out_dir}/local_extreme_vaf_{purity_tag}_{depth}.txt'

                if not os.path.isfile(var_tsv_path):
                    continue

                cmd = f'{script} vaf_hist_kde {kde_plot_path} {kde_result_path} {var_tsv_path} {kde_plot_title};'
                cmd += f'{script} find_local_extrema {kde_plot_path2} {local_extrema_txt_path} ' \
                       f'{kde_result_path} {var_tsv_path} {kde_plot_title};'

                if is_test:
                    print(cmd)
                else:
                    one_job_name = f'{job_name_prefix}.{cell_line}.{depth}.{purity_tag}'
                    prev_job_name = f'{prev_job_prefix}.{cell_line}.{depth}.{purity_tag}'
                    one_job = Job(one_job_name, cmd, hold_jid=prev_job_name)
                    jobs.append(one_job)

    if not is_test:
        qsub_sge(jobs, queue, log_dir)


def vaf_hist_kde(kde_plot_path, kde_result_path, var_tsv_path, kde_plot_title):
    """
    The function uses kernel density estimation for VAF histogram.
    The kernel density estimation smooths the shape of VAF histogram.

    :param kde_plot_path: a path of a KDE curve
    :param kde_result_path: a path of a KDE result
    :param var_tsv_path: a path of a TSV file for variants
    :param kde_plot_title: a title of the KDE plot
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
    bins = [round(0.01 * x, 2) for x in range(101)]

    for idx in range(len(bins) - 1):
        vaf_hist[(bins[idx], bins[idx + 1])] = 0  # initialize by 0

    for vaf in vaf_list:
        check = False  # check if data is counted

        for key in sorted(vaf_hist.keys(), key=lambda x: x[0]):
            if key[0] <= vaf < key[1]:  # binning for histogram
                vaf_hist[key] += 1
                check = True
                break

        if not check:  # if data is not counted, the data is 1.
            if vaf == 1.00:
                vaf_hist[(0.99, 1.00)] += 1
            else:
                sys.exit(f'[ERROR] Invalid VAF: {vaf}')

    eprint(f'[LOG] Draw KDE curve and histogram curve')
    fig, ax1 = plt.subplots()  # two independent plot(kde smooth hist and origin hist) are drawn at single figure.

    ax1.set_xlabel('VAF')
    ax1.set_ylabel('Density')  # kde density
    plt.title(kde_plot_title)

    kde_bandwidth = KDE_BANDWIDTH
    color = 'red'
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(vaf_list[:, numpy.newaxis])
    x_plot = numpy.linspace(0, 1, len(vaf_list))[:, numpy.newaxis]
    log_dens = kde.score_samples(x_plot)

    ax1.plot(x_plot[:, 0], numpy.exp(log_dens), color=color, label='bandwidth: %s' % kde_bandwidth)
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency')
    total_count_high_lodt = sum(vaf_hist.values())
    ax2.plot([round((x[0] + x[1]) / 2, 2) for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
             [vaf_hist[x] / total_count_high_lodt for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
             'g--', label='histogram (N: %s)' % total_count_high_lodt)

    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    fig.tight_layout()
    plt.savefig(kde_plot_path)
    plt.close()

    eprint(f'[LOG] Save KDE result')
    with open(kde_result_path, 'w') as kde_result_file:
        for x, y in zip(x_plot[:, 0], numpy.exp(log_dens)):
            kde_result_file.write(f'{x}\t{y}\n')


def find_local_extrema(out_plot_path, local_extrema_txt_path, kde_result_path, var_tsv_path, kde_plot_title):
    """
    The function finds local minima and local maxima from smoothed histogram.
    The results are going to be saved as a txt file and represented on a newly drawn plot.
    By finding local extrema, we can divide homozygous variants and heterozygous variants.
    """
    eprint(f'[LOG] Find local minima and maxima of KDE curve')
    kde_bandwidth = KDE_BANDWIDTH
    list_x = []
    list_kde = []

    with open(kde_result_path, 'r') as kde_file:
        for line in kde_file:
            cols = line.strip('\n').split('\t')
            list_x.append(float(cols[0]))
            list_kde.append(float(cols[1]))

    local_maxima = argrelextrema(numpy.array(list_kde), numpy.greater)  # find local maximum
    local_minima = argrelextrema(numpy.array(list_kde), numpy.less)  # find local minimum

    eprint('[LOG] Construct a VAF histrogram for plotting')
    vaf_list = []

    with open(var_tsv_path, 'r') as var_tsv_file:
        var_tsv_file.readline()  # remove a header

        for line in var_tsv_file:
            cols = line.strip().split('\t')
            vaf_list.append(float(cols[5]))

    vaf_list = numpy.array(vaf_list)
    vaf_hist = {}  # key: VAF bins, value: count; dictionary for histogram
    bins = [round(0.01 * x, 2) for x in range(101)]

    for idx in range(len(bins) - 1):
        vaf_hist[(bins[idx], bins[idx + 1])] = 0  # initialize by 0

    for vaf in vaf_list:
        check = False  # check if data is counted

        for key in sorted(vaf_hist.keys(), key=lambda x: x[0]):
            if key[0] <= vaf < key[1]:  # binning for histogram
                vaf_hist[key] += 1
                check = True
                break

        if not check:  # if data is not counted, the data is 1.
            if vaf == 1.00:
                vaf_hist[(0.99, 1.00)] += 1
            else:
                sys.exit(f'[ERROR] Invalid VAF: {vaf}')

    eprint(f'[LOG] Save local extrema as a text and represent them on the plot')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('VAF')
    ax1.set_ylabel('Density')
    plt.title(kde_plot_title)
    color = 'black'
    ax1.plot(list_x, list_kde, color=color, label='bandwidth: %s' % kde_bandwidth)

    vaf_to_label = {}  # key: a VAF, value: lmax (local maximum) or lmin (local minimum)

    for lmax_idx in local_maxima[0]:
        lmax_vaf = list_x[lmax_idx]
        vaf_to_label[lmax_vaf] = 'lmax'
        ax1.axvline(lmax_vaf, color='red', linestyle='--')
    for lmin_idx in local_minima[0]:
        lmin_vaf = list_x[lmin_idx]
        vaf_to_label[lmin_vaf] = 'lmin'
        ax1.axvline(list_x[lmin_idx], color='blue', linestyle='--')

    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency')
    total_count_high_lodt = sum(vaf_hist.values())
    ax2.plot([round((x[0] + x[1]) / 2, 2) for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
             [vaf_hist[x] / total_count_high_lodt for x in sorted(vaf_hist.keys(), key=lambda x:x[0])],
             'g--', label='histogram (N: %s)' % total_count_high_lodt)

    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(out_plot_path)
    plt.close()

    with open(local_extrema_txt_path, 'w') as local_extrema_txt_file:
        for vaf in sorted(vaf_to_label.keys()):
            print(vaf, vaf_to_label[vaf], sep='\t', file=local_extrema_txt_file)


def vaf_hist_kde_old(out_dir, ks_result_dir, diptest_result_path, silverman_result_path):
    """
    Deprecated function
    """
    '''
    we uses both of dip test and silverman test for bimodality test.
    silverman test is used for p value and dip test for VAF threshold.
    if silverman test's p value < 0.01, we decide the variants have bimodal distribution.

    '''
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
