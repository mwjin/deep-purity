#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make images for training or testing our CNN model

* Arguments of this script
    1. out_image_path = a path of a output image (that will be stored using pickle)
    2. var_tsv_file = a path of a file that contains contig, pos, ref, alt, vaf (this is a summary of MTO file)
    3. tumor_bam_path = a path of a tumor bam file
    4. normal_bam_path = a path of a normal bam file
    5. tumor purity (optional; only used for training and testing): a ratio of tumor samples
"""
import sys
import pysam
import numpy as np
import pickle
import random
import time

from decimal import Decimal


class ImageMaker(object):
    """
    a class for making images for our CNN
    """
    def __init__(self):
        """
        This initialization is called to make an image.
        """
        self.out_image_path = sys.argv[1]
        self.var_tsv_path = sys.argv[2]
        self.tumor_bam_path = sys.argv[3]
        self.norm_bam_path = sys.argv[4]

        # parameters for images
        self.allele_image_height = 50
        self.allele_image_width = 1000
        self.allele_image_channel_cnt = 9

        self.hist_height = 100
        self.hist_width = 101
        self.hist_channel_cnt = 1

        variants = self.load_variants()
        var_allele_image, vaf_hist_image = self.fetch_images(variants)
        tumor_purity = float(sys.argv[5]) if len(sys.argv) >= 5 else None  # used for training

        self.save_learn_info(var_allele_image, vaf_hist_image, tumor_purity)

    def load_variants(self):
        """
        :return: a list of '_Variant' objects with top 1000 LODt scores from {self.var_tsv_path} file
        """
        print('[LOG] Load variants')
        variants = _Variant.parse_tsv_file(self.var_tsv_path)

        if len(variants) < self.allele_image_width:  # over-sampling
            variants = list(np.random.choice(variants, self.allele_image_width))

        variants.sort(key=lambda variant: variant.lodt, reverse=True)
        variants = random.sample(variants, self.allele_image_width)

        return variants

    def fetch_images(self, variants):
        """
        Make an allele image of all variants and a VAF histogram image,
        and return both images (variant allele image, histogram image)
        """
        print('[LOG] Make an allele image for all variant positions', file=sys.stderr)
        var_allele_image = np.empty((self.allele_image_height, 0, self.allele_image_channel_cnt))
        vaf_list = []

        for variant in variants:
            vaf = variant.vaf
            vaf_list.append(vaf)

            tumor_alleles = self._get_alleles(self.tumor_bam_path, variant)
            normal_alleles = self._get_alleles(self.norm_bam_path, variant)

            var_allele_one_hot = ImageMaker._one_hot_encode(tumor_alleles, normal_alleles, variant)
            var_allele_one_hot = np.asarray(var_allele_one_hot).T
            var_allele_one_hot = var_allele_one_hot.reshape(self.allele_image_height, 1, self.allele_image_channel_cnt)
            var_allele_image = np.append(var_allele_image, var_allele_one_hot, axis=1)

        print('[LOG] Make a VAF histogram image')
        vaf_hist = dict()
        vaf_hist_interval = Decimal(1) / (self.hist_width - 1)

        for x in range(self.hist_width):
            vaf_hist[float(vaf_hist_interval * x)] = 0

        for vaf in vaf_list:
            vaf_hist[round(vaf, 2)] += 1

        vaf_hist_image = []  # a 2D list that represents the VAF histogram

        for vaf in sorted(vaf_hist.keys()):
            var_cnt = vaf_hist[vaf]
            vaf_col = [1.0] * self.hist_height
            zero_cnt = max(0, self.hist_height - var_cnt)

            for i in range(zero_cnt):
                vaf_col[i] = 0.0

            vaf_hist_image.append(vaf_col)

        vaf_hist_image = np.asarray(vaf_hist_image).T
        vaf_hist_image = vaf_hist_image[:, :, np.newaxis]

        return var_allele_image, vaf_hist_image

    def save_learn_info(self, var_allele_image, vaf_hist_image, tumor_purity=None):
        """
        Save a dictionary that contains information used for training or testing our learning model
        """
        print('[LOG] Save information for learning')

        learn_info_dict = {
            'var_allele_image': var_allele_image,
            'vaf_hist_image': vaf_hist_image,
            'tumor_purity': tumor_purity  # for training
        }

        with open(self.out_image_path, 'wb') as out_image_file:
            pickle.dump(learn_info_dict, out_image_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_alleles(self, bam_path, variant):
        """
        :return: a list of allele bases of a variant position
        """
        # get reads overlapped with the variant
        reads = ImageMaker._fetch_reads(bam_path, variant.chrom, variant.pos - 1)
        reads = list(filter(lambda x: x.query_position is not None, reads))

        if len(reads) < self.allele_image_height:
            reads = np.random.choice(reads, size=self.allele_image_height).tolist()
        else:
            reads = np.random.choice(reads, size=self.allele_image_height, replace=False).tolist()

        reads.sort(key=lambda x: x.alignment.mapping_quality, reverse=True)

        base_sort_prior_dict = dict()

        for base in ['A', 'C', 'G', 'T', 'N']:
            base_sort_prior_dict[base] = 1

        base_sort_prior_dict[variant.ref] = 2
        base_sort_prior_dict[variant.alt] = 0

        # get query bases of all reads
        alleles = []

        for pileup_read in reads:
            query_pos = pileup_read.query_position
            query_base = pileup_read.alignment.query_sequence[query_pos].upper()
            alleles.append(query_base)

        alleles.sort(key=lambda x: base_sort_prior_dict[x])
        return alleles

    @staticmethod
    def _fetch_reads(bam_path, contig, zero_based_pos):
        """
        Fetch reads with given a contig and position from a bam file
        :return:  a list of 'PileupRead' objects
        """
        raw_reads = []

        with pysam.AlignmentFile(bam_path, 'rb') as bam_file:
            # remove PCR duplicates and QC fail reads
            pileup_cols = bam_file.pileup(contig, zero_based_pos, zero_based_pos + 1, truncate=True, stepper='all')

            for pileup_column in pileup_cols:
                for pileup_read in pileup_column.pileups:
                    raw_reads.append(pileup_read)

        return raw_reads

    @staticmethod
    def _one_hot_encode(tumor_alleles, normal_alleles, variant):
        """
        :return: a 2D list that represents information of alleles (9 * self.allele_image_height)
        """
        # row: tumor_A, tumor_C, tumor_G, tumor_T, alt_allele, normal_A, normal_C, normal_G, normal_T
        # column: an one-hot encode of each base
        encode_result = []
        nts = ('A', 'C', 'G', 'T')

        # make rows for types of tumor alleles
        for nt in nts:
            tumor_base_binaries = []

            for base in tumor_alleles:
                if base == nt:
                    tumor_base_binaries.append(1.0)
                else:
                    tumor_base_binaries.append(0.0)

            encode_result.append(tumor_base_binaries)

        # make rows for alternative alleles in tumor alleles
        alt_allele_binaries = []

        for base in tumor_alleles:
            if base == variant.ref:
                alt_allele_binaries.append(0.0)
            elif base == variant.alt:
                alt_allele_binaries.append(1.0)
            else:
                alt_allele_binaries.append(0.5)  # very exceptional

        encode_result.append(alt_allele_binaries)

        # make rows for types of normal alleles
        for nt in nts:
            normal_base_binaries = []

            for base in normal_alleles:
                if base == nt:
                    normal_base_binaries.append(1.0)
                else:
                    normal_base_binaries.append(0.0)

            encode_result.append(normal_base_binaries)

        return encode_result


class _Variant:
    def __init__(self):
        self.chrom = '.'
        self.pos = 0  # 0-based
        self.ref = '.'
        self.alt = '.'
        self.lodt = 0.0
        self.vaf = 0.0

    def __str__(self):
        return f'{self.chrom}\t{self.pos}\t{self.ref}\t{self.alt}\t{self.lodt}\t{self.vaf}'

    def parse_tsv_entry(self, tsv_entry):
        # TSV cols: chrom, pos, ref, alt, LODt score, VAF
        tsv_fields = tsv_entry.strip().split('\t')
        self.chrom = tsv_fields[0]
        self.pos = int(tsv_fields[1])
        self.ref = tsv_fields[2]
        self.alt = tsv_fields[3]
        self.lodt = float(tsv_fields[4])
        self.vaf = float(tsv_fields[5])

    @staticmethod
    def parse_tsv_file(tsv_file_path):
        variants = []

        with open(tsv_file_path, 'r') as tsv_file:
            tsv_file.readline()  # remove a header

            for tsv_entry in tsv_file:
                variant = _Variant()
                variant.parse_tsv_entry(tsv_entry)
                variants.append(variant)

        return variants


if __name__ == '__main__':
    print(time.ctime(), file=sys.stderr)
    ImageMaker()
    print(time.ctime(), file=sys.stderr)
