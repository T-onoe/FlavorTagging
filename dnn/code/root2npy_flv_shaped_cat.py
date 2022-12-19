import ROOTTOOLS
import TOOLS
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


if __name__ == "__main__":
        pdpath = "/home/onoe/ILC/Deep_Learning/data/npy/1mpure/categories/v01-06/"
        npyname_s = pdpath + "Pn23n23h_cc_sum_shaped.npy"
        npyname = pdpath + "Pn23n23h_cc_sum.npy"
        fnames = ["/home/onoe/ILC/Deep_Learning/data/ILD_sample_250_01/makentuple_Pn23n23h_cc_sum.root"]
        now = datetime.datetime.now()
        
        bnames = ['nvtx', 'nvtxall', '1vtxprob',
                  'vtxlen1_jete', 'vtxlen2_jete', 'vtxlen12_jete',
                  'vtxsig1_jete', 'vtxsig2_jete', 'vtxsig12_jete',
                  'vtxdirang1_jete', 'vtxdirang2_jete', 'vtxdirang12_jete',
                  'vtxmom_jete', 'vtxmom1_jete', 'vtxmom2_jete',
                  'vtxmass', 'vtxmass1', 'vtxmass2', 'vtxmasspc',
                  'vtxmult', 'vtxmult1', 'vtxmult2', 'vtxprob',
                  'trk1d0sig', 'trk2d0sig',
                  'trk1z0sig', 'trk2z0sig',
                  'trk1pt_jete', 'trk2pt_jete',
                  'jprobr2', 'jprobz2',
                  'jprobr25sigma', 'jprobz25sigma', 'trkmass',
                  'nmuon', 'nelectron',
                  'd0bprob2', 'd0cprob2', 'd0qprob2',
                  'z0bprob2', 'z0cprob2', 'z0qprob2', 
                   ]                  

        data, nevent = ROOTTOOLS.fileload_setnames(fnames, "ntp", bnames)
        data = np.array(data).reshape(nevent, len(bnames))
        
#nvtx 1vtxprob
        data[:, 0] = TOOLS.0filter(data[:, 0])
        data[:, 1] = TOOLS.rescaling(data[:, 1])
        data[:, 2] = TOOLS.logarithm3(data[:, 2])
#vtxlen
        data[:, 3] = TOOLS.logarithm4(data[:, 3])
        data[:, 4] = TOOLS.logarithm2(data[:, 4])
        data[:, 5] = TOOLS.logarithm3(data[:, 5])
#vtxsig_jete
        data[:, 6] = TOOLS.logarithm5(data[:, 6])
        data[:, 7] = TOOLS.logarithm2(data[:, 7])
        data[:, 8] = TOOLS.logarithm6(data[:, 8])
#vtxdirang_jete 
        data[:, 9] = TOOLS.logarithm2(data[:, 9])
        data[:, 10] = TOOLS.logarithm2(data[:, 10])
        data[:, 11] = TOOLS.logarithm2(data[:, 11])
#vtxmom_jete
        data[:, 12] = TOOLS.logarithm2(data[:, 12])
        data[:, 13] = TOOLS.logarithm2(data[:, 13])
        data[:, 14] = TOOLS.logarithm7(data[:, 14])
#vtxmass
        data[:, 15] = TOOLS.logarithm1(data[:, 15])
        data[:, 16] = TOOLS.logarithm1(data[:, 16])
        data[:, 17] = TOOLS.logarithm1(data[:, 17])
        data[:, 18] = TOOLS.logarithm1(data[:, 18])
#vtxmult
#trksig
        data[:, 23] = abs(data[:, 23])
        data[:, 23] = TOOLS.logarithm1(data[:, 23])
        data[:, 24] = abs(data[:, 24])
        data[:, 24] = TOOLS.logarithm1(data[:, 24])
        data[:, 25] = abs(data[:, 25])
        data[:, 25] = TOOLS.logarithm3(data[:, 25])
        data[:, 26] = abs(data[:, 26])
        data[:, 26] = TOOLS.logarithm3(data[:, 26])
#trkpt_jete
        data[:, 27] = TOOLS.logarithm3(data[:, 27])
        data[:, 28] = TOOLS.logarithm3(data[:, 28])
#jprob2
        data[:, 29] = TOOLS.logarithm3(data[:, 29])
        data[:, 30] = TOOLS.logarithm3(data[:, 30])
#jprob25sigma
        data[:, 31] = TOOLS.logarithm1(data[:, 31])
        data[:, 32] = TOOLS.logarithm1(data[:, 32])
#trkmass
        data[:, 33] = TOOLS.logarithm1(data[:, 33])
#*prob2
        data[:, 36] = TOOLS.logarithm(data[:, 36])
        data[:, 37] = TOOLS.logarithm(data[:, 37])
        data[:, 38] = TOOLS.logarithm(data[:, 38])
        data[:, 39] = TOOLS.logarithm(data[:, 39])
        data[:, 40] = TOOLS.logarithm(data[:, 40])
        data[:, 41] = TOOLS.logarithm(data[:, 41])

#remove nan        
        data_shaped = data[~np.isnan(data).any(axis=1), :]
        
        print("saving...")
        np.save(npyname, data, fix_imports=True)
