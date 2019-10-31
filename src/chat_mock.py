#!/usr/bin/env python3
"""
Estimate purity via CHAT algorithm (Li and Li., Genome Biology, 2014)
"""
import sys
import numpy as np
import pandas as pd
import math

from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def main():
    # path settings
    in_seg_tsv_path = sys.argv[2]
    output_path = sys.argv[1]

    # param setting
    max_ploidy = 8
    npl = 2

    print('[LOG] Estimate a tumor purity using the genome segments')
    segments = get_segments(in_seg_tsv_path)
    origin, is_euploid = get_origin(segments)
    non_euploid_segments = segments[np.invert(is_euploid)]
    non_euploid_segments = segment_filter(non_euploid_segments, origin)
    min_square_dists, est_purities = get_min_square_dists(non_euploid_segments, max_ploidy, npl, origin)
    updated_segments = append_info_to_segments(non_euploid_segments, min_square_dists, est_purities)

    print('[LOG] Write the result of the mocked CHAT')
    with open(output_path, 'w') as outfile:
        print('folded_VAF', 'log2-LRR', 'min_square_dist', 'est_purity', sep='\t', file=outfile)

        for segment in updated_segments:
            print(*segment, sep='\t', file=outfile)


def get_segments(seg_path):
    print('[LOG] ... Read the Rdata file and get segments')
    seg_df = pd.read_table(seg_path)
    seg_df = seg_df.dropna()

    folded_bafs = seg_df['folded_VAF'].values
    log_lrrs = seg_df['log2-LRR'].values

    return np.asarray(list(zip(folded_bafs, log_lrrs)))


def get_origin(segments):
    print('[LOG] ... Get the euploid origin of the segments')
    euploid_cl_cnt = np.zeros(len(segments))

    print('[LOG] ...... Find the cluster with euploid segments using K-means clustering')

    for i in range(10):
        print(f'[LOG] ......... No.{i + 1} K-means clustering')
        cluster_model = chat_kmeans(segments)
        cl_centers = cluster_model.cluster_centers_
        cl_labels = cluster_model.labels_

        center_norms = np.linalg.norm(cl_centers, axis=1)
        euploid_label = np.argmin(center_norms)
        euploid_cl_cnt[cl_labels == euploid_label] += 1

    # origin after K-means clustering
    is_euploid = euploid_cl_cnt >= 6
    origin_segs = segments[is_euploid]
    origin = np.mean(origin_segs, axis=0)

    std_baf = 0.01
    std_lrr = 0.4

    print('[LOG] ...... Pull other non-euploid segments into the euploid cluster')
    for i, segment in enumerate(segments):
        if not is_euploid[i]:
            if abs(segment[0] - origin[0]) <= std_baf and abs(segment[1] - origin[1]) <= std_lrr:
                is_euploid[i] = True

    origin_segs = segments[is_euploid]
    origin = np.mean(origin_segs, axis=0)

    return origin, is_euploid


def chat_kmeans(segments):
    # Get an origin of the segments
    baf_max, lrr_max = np.squeeze((np.max(segments, axis=0)))
    baf_min, lrr_min = np.squeeze((np.min(segments, axis=0)))
    weight = (lrr_max - lrr_min) / (baf_max - baf_min)

    weighted_segs = deepcopy(segments)
    weighted_segs[:, 0] *= weight
    num_clusters = 2

    # clustering using k where the inertia is lower than our threshold
    while True:
        cluster_model = KMeans(n_clusters=num_clusters, max_iter=10, n_init=50).fit(weighted_segs)
        inertia_thr = 0.1 * weight * num_clusters

        if cluster_model.inertia_ >= inertia_thr:
            num_clusters += 1
        else:
            break

    cl_centers = cluster_model.cluster_centers_
    cl_centers /= weight

    return cluster_model


def segment_filter(segments, origin):
    x0 = origin[0]
    y0 = origin[1]

    purities = np.arange(0.0, 1.01, 0.01)
    base_lrrs = ((2 * 2 * (1 - purities) + 1 * purities) / (2 * 2))
    base_lrrs = np.array(list(map(math.log2, base_lrrs))) + y0
    base_bafs = (0 * purities + 2 * (1 - purities)) / (2 * 2 * (1 - purities) + 1 * purities)
    base_bafs = np.array(list(map(abs, base_bafs - 0.5))) + x0

    filt_seg_points = []

    for seg_x, seg_y in segments:
        min_idx = np.argmin(abs(base_bafs - seg_x))

        if base_lrrs[min_idx] - seg_y < 0.5:
            filt_seg_points.append((seg_x, seg_y))

    return np.array(filt_seg_points)


def get_min_square_dists(segments, max_ploidy, npl, origin):
    """
    Return the min squared dists of all segments and their cognate estimated purities
    """
    purities = np.arange(0.01, 1.01, 0.01)
    min_square_dists = np.full(len(segments), 10000, dtype='float64')
    est_purities = np.full(len(segments), 10000, dtype='float64')
    origin_x, origin_y = origin

    for est_purity in purities:
        canonical_lines = get_canonical_lines(origin_x, origin_y, max_ploidy, npl, est_purity)
        square_dists = get_square_dists(segments, canonical_lines)
        est_purities[square_dists < min_square_dists] = est_purity
        min_square_dists[square_dists < min_square_dists] = square_dists[square_dists < min_square_dists]

    return min_square_dists, est_purities


def get_canonical_lines(x0, y0, max_ploidy, npl, purity=1.0):
    canonical_lines = []  # item: (coef, intercept)
    for nb in range(int(max_ploidy / 2)):
        canonical_points = get_canonical_points(x0, y0, nb, max_ploidy, npl, purity)
        canonical_linear_model = \
            LinearRegression().fit(canonical_points[:, 0].reshape(-1, 1), canonical_points[:, 1])
        canonical_lines.append((canonical_linear_model.coef_[0], canonical_linear_model.intercept_))

    return canonical_lines


def get_canonical_points(x0, y0, nb, max_ploidy, npl, purity=1.0):
    """
    Reference: CHAT supplementary information
    """
    canonical_points = []  # item: (|BAF - 0.5|, log2(LRR))

    if (nb * 2) <= max_ploidy:
        for nt in range((nb * 2), max_ploidy + 1):
            if nt == 0:
                continue

            simulated_nb = purity * nb + (1 - purity) * npl
            simulated_nt = purity * nt + (1 - purity) * 2 * npl

            baf = simulated_nb / simulated_nt
            lrr = simulated_nt / (2 * npl)

            folded_baf = abs(baf - 0.5) + x0
            log_lrr = math.log2(lrr) + y0
            canonical_points.append((folded_baf, log_lrr))

    return np.asarray(canonical_points)


def get_square_dists(points, canonical_lines):
    point_cnt = len(points)
    min_dists = np.full(point_cnt, fill_value=10000, dtype='float64')

    for canonical_coef, canonical_intercept in canonical_lines:
        for i, point in enumerate(points):
            point_x, point_y = point

            if canonical_coef == 0.0:  # purity == 1.0, nb = 0
                dist = get_line_point_dist(point_x, point_y, -1, 0, 0.5)
            else:
                dist = get_line_point_dist(point_x, point_y, canonical_coef, -1, canonical_intercept)

            if dist < min_dists[i]:
                min_dists[i] = dist

    return min_dists ** 2


def get_line_point_dist(x, y, x_coef, y_coef, intercept):
    return abs(x * x_coef + y * y_coef + intercept) / np.linalg.norm((x_coef, y_coef))


def append_info_to_segments(segments, min_square_dists, est_purities):
    """
    append (min_square_dist, est_purity) used to estimate the tumor purity to each segment
    """
    updated_segments = []

    for i in range(len(segments)):
        segment = list(segments[i])
        min_square_dist = min_square_dists[i]
        est_purity = est_purities[i]
        segment.append(min_square_dist)
        segment.append(est_purity)
        updated_segments.append(segment)

    return np.array(updated_segments)


if __name__ == '__main__':
    main()
