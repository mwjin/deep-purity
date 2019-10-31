#!/usr/bin/env Rscript
# Title     : segmentation.R
# Objective : Segmentation of a genome like CHAT (Li and Li., Genome Biology, 2014)
# Created by: Minwoo Jeong
# Created on: 2019-08-05
# References: https://rdrr.io/cran/CHAT/src/R/getSegChr.R, https://rdrr.io/cran/CHAT/src/R/getBAFmean.R

main <- function() {
    # parse arguments
    args = commandArgs(TRUE)
    outfile_path = args[1]
    var_tsv_path = args[2]

    # param settings
    seg_size = as.numeric(args[3])

    variant_df = read.table(var_tsv_path, sep='\t', header=T, stringsAsFactors=F)
    variant_df = variant_df[variant_df['judgement'] == 'REJECT',]  # only germline varaints

    seg_df = c()
    chroms = unique(variant_df[,'contig'])

    # segmentation
    for (chrom in chroms) {
        chr_variant_df = variant_df[variant_df['contig'] == chrom,]
        var_cnt = nrow(chr_variant_df)
        num_iter = ceiling(var_cnt / seg_size)

        for (i in 1:num_iter) {
            seg_start_idx = (i - 1) * seg_size + 1
            seg_end_idx = i * seg_size

            if (seg_end_idx > var_cnt)
                seg_end_idx = var_cnt

            seg_variant_df = chr_variant_df[seg_start_idx:seg_end_idx,]

            t_ref_count = as.numeric(unlist(seg_variant_df['t_ref_count']))
            t_alt_count = as.numeric(unlist(seg_variant_df['t_alt_count']))
            t_depth = t_ref_count + t_alt_count

            n_ref_count = as.numeric(unlist(seg_variant_df['n_ref_count']))
            n_alt_count = as.numeric(unlist(seg_variant_df['n_alt_count']))
            n_depth = n_ref_count + n_alt_count

            log2_lrr = log2(t_depth / n_depth)
            seg_folded_vaf = get_seg_folded_vaf(unlist(seg_variant_df['tumor_f']))
            seg_log2_lrr = median(log2_lrr)

            real_seg_size = seg_end_idx - seg_start_idx + 1
            seg_df = rbind(seg_df, c(chrom, chr_variant_df[seg_start_idx, 'position'], chr_variant_df[seg_end_idx, 'position'], seg_folded_vaf, seg_log2_lrr, real_seg_size))
        }
    }
    colnames(seg_df) = c('chrom', 'start', 'end', 'folded_VAF', 'log2-LRR', 'size')
    write.table(seg_df, file=outfile_path, sep='\t', row.names=F, quote=F)
}

get_seg_folded_vaf <- function(vafs) {
    # this is a modified version of the function from https://rdrr.io/cran/CHAT/src/R/getBAFmean.R
    if (length(vafs) < 10)
        return(NA)

    vaf_quantile = quantile(vafs, na.rm=T, c(0.01, 0.99))
    vafs = vafs[vafs >= vaf_quantile[1] & vafs <= vaf_quantile[2]]

    if (length(vafs) < 10)
        return(NA)

    x = density(vafs, na.rm=T)$x
    y = density(vafs, na.rm=T)$y

    # non-linear regression to the bimodal distribution (the sum of two normal distributions)
    # and the unimodal distribution (the normal distribution)
    model_std = c(0.06, 0.06)
    model_mean = c(0.05, 0.95)
    model_vars = list(a=1 / (sqrt(2 * pi) * model_std[1]), b=model_std[1], c=model_mean[1], d=1 / (sqrt(2 * pi) * model_std[2]), e=model_std[2], f=model_mean[2])

    # try until no error
    while(1) {
        bi_model = try(nls(y ~ a / b * exp(-(x - c) ^ 2 / (2 * b ^ 2)) + d / e * exp(-(x - f) ^ 2 / (2 * e ^ 2)), start=model_vars, trace=F), silent=T)

        if (class(bi_model) == 'nls')
            break

        model_mean = model_mean + c(0.05, -0.05)

        if (abs(model_mean[1] - model_mean[2]) <= 0.001)
            break

        model_vars = list(a=1 / (sqrt(2 * pi) * model_std[1]), b=model_std[1], c=model_mean[1], d=1 / (sqrt(2 * pi) * model_std[2]), e=model_std[2], f=model_mean[2])
    }

    uni_model = try(nls(y ~ 1 / (sqrt(2 * pi) * a) * exp(-(x - b) ^ 2 / (2 * a ^ 2)), start=list(a = 0.06, b = 0.5)), silent=T)

    if (class(uni_model) == 'nls' & class(bi_model) == 'nls') {
        uni_model_summary <- summary(uni_model)
        bi_model_summary <- summary(bi_model)
        bi_model_mean1 = bi_model_summary$parameters[3, 1]
        bi_model_mean2 = bi_model_summary$parameters[6, 1]

        if (abs(bi_model_mean1 - 0.5) < 0.08 | abs(bi_model_mean2 - 0.5) < 0.08) {
            if (abs(bi_model_mean1 - 0.5) < 0.08 & abs(bi_model_mean2 - 0.5) < 0.08)
                return(abs(bi_model_mean2 - bi_model_mean1) / 2)
            else {
                if (abs(bi_model_mean1 - 0.5) + abs(bi_model_mean2 - 0.5) > 0.2)
                    return(0)
                else
                    return(abs(bi_model_mean2 - bi_model_mean1) / 2)
            }
        }
        else {
            if (uni_model_summary$sigma < 3 * bi_model_summary$sigma)
                return(0)

            if (bi_model_mean1 >= bi_model_mean2) {
                small_mean = bi_model_mean2
                large_mean = bi_model_mean1
            }
            else {
                small_mean = bi_model_mean1
                large_mean = bi_model_mean2
            }

            if (small_mean < 0 | large_mean > 1 | small_mean + large_mean > 1.2)
                return(NA)

            return(abs(bi_model_mean2 - bi_model_mean1) / 2)
        }
    }
    else if (class(bi_model) == 'nls') {
        bi_model_summary <- summary(bi_model)
        bi_model_mean1 = bi_model_summary$parameters[3, 1]
        bi_model_mean2 = bi_model_summary$parameters[6, 1]
        return(abs(bi_model_mean2 - bi_model_mean1) / 2)
    }
    else if (class(uni_model) == 'nls')
        return(0)
    else
        return(NA)
}

if (!interactive()) {
    main()
}
