#=======================================================================================================#
#===ROOT TOOLS FOR DATA ANALYSIS========================================================================#
#=======================================================================================================#
import numpy as np
from tqdm import tqdm

import ROOT


def fileread(fname, tname):
    f = ROOT.TFile(fname, "read")
    tree = f.Get(tname)
    
    lb = tree.GetListOfBranches()
    bnames = [lb.At(s).GetName() for s in range(lb.GetEntries())]
    
    return f, tree, bnames


def fileload(fnames, tname):
    data = []
    ievent = 0
    
    for fname in fnames:
        f, tree, bnames = fileread(fname, tname)
        
        for i in tqdm(range(tree.GetEntries())):
            tree.GetEntry(i)
            if tree.connect<0: continue
            data.append([])
            data[ievent].append([getattr(tree, bname) for bname in bnames])
            ievent += 1
            
    return data, ievent


def fileload_setnames(fnames, tname, bnames):
    data = []
    ievent = 0

    for fname in fnames:
        f, tree, _ = fileread(fname, tname)
        
        #for i in tqdm(range(tree.GetEntries())):
        for i in tqdm(range(100000)):
            tree.GetEntry(i)
            #if tree.connect<0: continue
            data.append([])
            data[ievent].append([getattr(tree, bname) for bname in bnames])
            ievent += 1

    return data, ievent 

