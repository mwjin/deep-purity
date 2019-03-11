#!/extdata6/Beomman/bins/anaconda3/bin/python

import sys
import pysam
import numpy
import pickle

class MakeCNNImage(object):

    def __init__(self):
        
        self.tumor_ = sys.argv[1]
        self.normal_ = sys.argv[2]
        self.mto_ = sys.argv[3]
        self.sample_name = sys.argv[4]
        self.width = int(sys.argv[5])
        self.hist_height = int(sys.argv[6])
        
        dic_vaf = self.load_vaf_from_mto(self.mto_)

        list_path = self.mto_to_position_file(self.mto_)
        for idx, path_ in enumerate(list_path):
            list_pos = self.load_pos_file(path_)
            dic_array = self.fetch_image_arrays(list_pos, self.tumor_, self.normal_, dic_vaf)
            
            if dic_array['read_image'].shape != (50, self.width, 9):
                print(dic_array['read_image'].shape)
                sys.exit()
            if dic_array['vaf_hist_image'].shape != (self.hist_height, 101, 1):
                print(dic_array['vaf_hist_image'].shape)
                sys.exit()

            out_pkl_ = '../%s/images/%s_image_%s.pkl' %(self.sample_name, self.sample_name, idx)
            self.dump_array(dic_array, self.out_pkl_)

        return

    def load_mto_file(self, mto_):
    
        normal_contigs = ['%d' %(contig+1) for contig in range(22)] + ['X', 'Y']
        list_variant = []
        
        mto = open(mto_, 'r')
        mto_ver =  mto.readline().strip('\n')
        header = mto.readline().strip('\n').split('\t')
        
        contig_idx = header.index('contig')
        pos_idx = header.index('position')
        ref_idx = header.index('ref_allele')
        alt_idx = header.index('alt_allele')
        lodt_idx = header.index('t_lod_fstar')
        judgement_idx = header.index('judgement')
        
        for line in mto:
            cols = line.strip('\n').split('\t')
            contig = cols[contig_idx]
            if contig not in normal_contigs:
                continue
            one_based_pos = int(cols[pos_idx])
            ref = cols[ref_idx]
            alt = cols[alt_idx]
            lodt = float(cols[lodt_idx])
            judgement = cols[judgement_idx]
            list_variant.append((contig, one_based_pos, ref, alt, lodt, judgement))
        mto.close()

        return list_variant

    def mto_to_position_file(self, mto_):
        
        list_path = []
        list_variant = load_mto_file(mto_)
        list_pass = [x for x in list_variant if x[6] == 'KEEP']
        for idx in range(1000):
            if len(list_pass) > 10000:
                list_sampled_idx = numpy.random.choice(range(len(list_pass)), size=10000, replace=False)
                list_mutect_sampled = [ list_pass[x] for x in list_sampled_idx ]
            else:
                list_sampled_idx = numpy.random.choice(range(len(list_pass)), size=10000)
                list_mutect_sampled = [ list_pass[x] for x in list_sampled_idx ]
                
            list_mutect_sampled = sorted(list_mutect_sampled, key=lambda x:x[4], reverse=True)[:1000]
            
            out_ = '../%s/pos/%s_positions_%s.tsv' %(self.sample_name, self.sample_name, idx)
            list_path.append(out_)
            out = open(out_, 'w')
            for x in list_mutect_sampled:
                out.write('%s\t%s\t%s\t%s\t%s\t%s\n' %(x[0],x[1],x[2],x[3],x[4],x[5]))
            out.close()

        return list_path

    def dump_array(self, dic_array, out_pkl_):
        
        out_pkl = open(out_pkl_, 'wb')
        pickle.dump(dic_array, out_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        out_pkl.close()
    
        return
    
    def load_vaf_from_mto(self, mto_):
        
        dic_vaf = dict() ## {0.0:1, ...}
        
        mto = open(mto_, 'r')
        mto_ver = mto.readline().strip('\n').split('\t')
        overhead = mto.readline().strip('\n').split('\t')
        contig_idx = overhead.index('contig')
        pos_idx = overhead.index('position')
        vaf_idx = overhead.index('tumor_f')
        for line in mto:
            cols = line.strip('\n').split('\t')
            
            contig = cols[contig_idx]
            pos = int(cols[pos_idx])
            vaf = float(cols[vaf_idx])

            dic_vaf[(contig, pos)] = vaf
        mto.close()
        
        return dic_vaf

    def load_pos_file(self, pos_):
        
        list_pos = []
        pos_file = open(pos_, 'r')
        for line in pos_file:
            cols =  line.strip('\n').split('\t')
            contig = cols[0]
            zero_based_pos = int(cols[1])-1
            ref = cols[2]
            alt = cols[3]
            list_pos.append((contig, zero_based_pos, ref, alt))
        pos_file.close()
        
        return list_pos

    def fetch_raw_reads_from_bam(self, tumor_, contig, zero_based_pos):
        
        list_raw_read = []
        tumor_data = pysam.AlignmentFile(tumor_, 'rb')
        for pileupcolumn in tumor_data.pileup(contig, zero_based_pos, zero_based_pos+1, truncate=True, stepper='all'): ## filter pcr dup, qc fail, ...
            for pileupread in pileupcolumn.pileups:
                list_raw_read.append(pileupread)
        tumor_data.close()

        return list_raw_read

    def fetch_image_arrays(self, list_pos, tumor_, normal_, dic_vaf):
        
        list_array = []
        list_vaf = []
        for pos in list_pos:
            contig = pos[0]
            zero_based_pos = pos[1]
            vaf = dic_vaf[(contig, zero_based_pos+1)]
            list_vaf.append(vaf)

            list_tumor_read = self.fetch_raw_reads_from_bam(tumor_, contig, zero_based_pos)
            list_tumor_read = [ x for x in list_tumor_read if x.query_position != None ]

            if len(list_tumor_read) > 0 and len(list_tumor_read) < 50:
                list_tumor_read = numpy.random.choice(list_tumor_read, size=50)
            elif len(list_tumor_read) > 50:
                list_tumor_read = numpy.random.choice(list_tumor_read, size=50, replace=False)

            list_tumor_read = sorted(list_tumor_read, key=lambda x:x.alignment.mapping_quality, reverse=True)

            list_tumor_base = []
            for pileupread in list_tumor_read:
                query_pos = pileupread.query_position
                if query_pos == None:
                    list_tumor_base.append('D')
                    continue

                query_base = pileupread.alignment.query_sequence[query_pos].upper()
                list_tumor_base.append(query_base)

            dic_for_sorting = {pos[2]:0, pos[3]:2, 'N':1}
            for nt in ['A','C','G','T']:
                if nt != pos[2] and nt != pos[3]:
                    dic_for_sorting[nt] = 1

            list_tumor_base = sorted(list_tumor_base, key=lambda x:dic_for_sorting[x], reverse=True)

            list_normal_read = self.fetch_raw_reads_from_bam(normal_, contig, zero_based_pos)
            list_normal_read = [ x for x in list_normal_read if x.query_position != None ]

            if len(list_normal_read) > 0 and len(list_normal_read) < 50:
                list_normal_read = numpy.random.choice(list_normal_read, size=50)
            elif len(list_normal_read) > 50:
                list_normal_read = numpy.random.choice(list_normal_read, size=50, replace=False)

            list_normal_read = sorted(list_normal_read, key=lambda x:x.alignment.mapping_quality, reverse=True)

            list_normal_base = []
            for pileupread in list_normal_read:
                query_pos = pileupread.query_position
                if query_pos == None:
                    list_normal_base.append('D')
                    continue

                query_base = pileupread.alignment.query_sequence[query_pos].upper()
                list_normal_base.append(query_base)

            list_normal_base = sorted(list_normal_base, key=lambda x:dic_for_sorting[x], reverse=True)

            if len(list_tumor_base) == 0:
                list_tumor_base = [ 'X' for x in range(50) ]
            if len(list_normal_base) == 0:
                list_normal_base = [ 'X' for x in range(50) ]

            list_one_hot_encoding_array = []
            nts = ('A', 'C', 'G', 'T')
            for nt in nts:
                list_nt = []
                #for base in list_base:
                for idx in range(50):
                    base = list_tumor_base[idx]
                    if base == nt:
                        list_nt.append(1.0)
                    else:
                        list_nt.append(0.0)
                list_one_hot_encoding_array.append(list_nt)

            list_ref_alt = []
            for base in list_tumor_base:
                if base == pos[2]:
                    list_ref_alt.append(0.0)
                elif base == pos[3]:
                    list_ref_alt.append(1.0)
                else:
                    list_ref_alt.append(0.5)
            list_one_hot_encoding_array.append(list_ref_alt)

            for nt in nts:
                list_nt = []
                for idx in range(50):
                    base = list_normal_base[idx]
                    if base == nt:
                        list_nt.append(1.0)
                    else:
                        list_nt.append(0.0)
                list_one_hot_encoding_array.append(list_nt)

            list_array.append(list_one_hot_encoding_array)

        #list_array2 = numpy.zeros((50, 100, 9))
        #list_array2 = numpy.zeros((50, 1000, 9))
        list_array2 = numpy.zeros((50, self.width, 9))
        for idx, data_point in enumerate(list_array):
            for idx2, channel in enumerate(data_point):
                for idx3, ele in enumerate(channel):
                    list_array2[idx3][idx][idx2] += ele

        dic_vaf = dict()
        for x in range(101):
            dic_vaf[round(0.01*x, 2)] = 0

        for vaf in list_vaf:
            dic_vaf[round(vaf, 2)] += 1
        
        total_count = sum(dic_vaf.values())
        for key in dic_vaf.keys():
            dic_vaf[key] = dic_vaf[key]/total_count

        list_vaf2 = []
        for vaf in sorted(dic_vaf.keys()):
            list_vaf2.append((vaf, dic_vaf[vaf]))
        
        #print(list_vaf2)
        #print(sum([x[1] for x in list_vaf2]))

        list_vaf_image = []
        for vaf in list_vaf2:
            col = []
            #for idx in range(100-int(vaf[0]*100)):
            zero_idx = max(0, self.hist_height-int(vaf[1]*1000))
            for idx in range(zero_idx):
                col.append(0.0)
            #for idx in range(int(vaf[0]*100)):
            one_idx = min(self.hist_height, int(vaf[1]*1000))
            for idx in range(one_idx):
                col.append(1.0)
            list_vaf_image.append(col)

        list_vaf_image = numpy.array(list_vaf_image).T
        list_vaf_image = list_vaf_image[:,:,numpy.newaxis]
        #list_vaf3 = sorted(list_vaf2, key=lambda x:x[1], reverse=True)

        dic_array = {'read_image':list_array2, 'vaf_hist_image':list_vaf_image}

        return dic_array

MakeCNNImage()
