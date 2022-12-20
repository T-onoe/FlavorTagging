import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import itertools

class TrkDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)

    @property
    def raw_file_names(self):
        for i in range(10,100):
            flavor_list = ['bb', 'cc', 'uudd']
            for flavor in flavor_list:
                trk_all = np.load("/gluster/maxi/ilc/onoe/gnn/data_1017/npy/" + flavor + "/trk/{}000.npy".format(i))
                vtx_all = np.load("/gluster/maxi/ilc/onoe/gnn/data_1017/npy/" + flavor + "/vtx/{}000.npy".format(i))
                return trk_all, vtx_all
            
    @property
    def processed_file_names(self):
        DIR = '/gluster/maxi/ilc/onoe/gnn/data_1212/processed'
#        idx_max = sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR))
        idx_max = 10000
        return ['/gluster/maxi/ilc/onoe/gnn/data_1212/processed/' + f'data_{i}.pt' for i in range(idx_max)]

    def process(self):
        flavor_list = ['bb', 'cc', 'uudd']
        counter = -1
        for file_num in tqdm(range(10,100)):
            for flavor in flavor_list:
                print("----- " + flavor + " jet process start. -----")
                trk_all = np.load("/gluster/maxi/ilc/onoe/gnn/data_1017/npy/" + flavor + "/trk/{}000.npy".format(file_num))
                vtx_all = np.load("/gluster/maxi/ilc/onoe/gnn/data_1017/npy/" + flavor + "/vtx/{}000.npy".format(file_num))
                nev = int(np.amax(trk_all[:,0]))
                njet = int(np.amax(trk_all[:,1]))
                # Max trk,vtx per a jet
                trk_max = int(np.amax(trk_all[:,2])) + 1
                vtx_max = int(np.amax(vtx_all[:,2])) + 1
                index_max = trk_max + vtx_max
                #行数取得
                trk_line = trk_all.shape[0]
                vtx_line = vtx_all.shape[0]
                
                nev_trk, index_trk, trk_npy, trk_chi2 = np.split(trk_all, [2,3,8], 1)
                nev_vtx, index_vtx, adj_trks, vtx_npy, trueVtx = np.split(vtx_all, [2,3,5,9], 1)
                
                for ev in tqdm(range(nev)):
                    for j in range(njet):
                        #Track features
                        trk_2d = np.empty((1, 5))
                        label = np.empty((1,1))
                        for nline_trk in range(trk_line):
                            if nev_trk[nline_trk,0]==ev and nev_trk[nline_trk,1]==j:
                                target = np.reshape(trk_npy[nline_trk,:],[1, 5])
                                trk_2d = np.concatenate([trk_2d, target], axis=0)
                                
                        extra, trk_2d = np.split(trk_2d, [1], 0)

                        #Track origin label
                        num_trk_2d = trk_2d.shape[0]
                        label = np.zeros(num_trk_2d)
                        for nline_vtx in range(vtx_line):
                            if nev_vtx[nline_vtx,0]==ev and nev_vtx[nline_vtx,1]==j:
                                if trueVtx[nline_vtx,:]== 1:
                                    pv1, pv2 = np.split(adj_trks[nline_vtx,:], 2)
                                    label[int(pv1)]=1
                                    label[int(pv2)]=1
                                elif trueVtx[nline_vtx,0]==2:
                                    svb1, svb2 = np.split(adj_trks[nline_vtx,:], 2)
                                    label[int(svb1)]=2
                                    label[int(svb2)]=2
                                elif (trueVtx[nline_vtx,0]==3) & (flavor == 'bb'):
                                    tvc1, tvc2 = np.split(adj_trks[nline_vtx,:], 2)
                                    label[int(tvc1)]=3
                                    label[int(tvc2)]=3
                                elif (trueVtx[nline_vtx,0]==3) & (flavor == 'cc'):
                                    svc1, svc2 = np.split(adj_trks[nline_vtx,:], 2)
                                    label[int(svc1)]=4
                                    label[int(svc2)]=4

                        #Node-level + Graph-level label
                        if flavor == 'bb':
                            graph_label = np.array([[0]])
                        elif flavor == 'cc':
                            graph_label = np.array([[1]])
                        elif flavor == 'uudd':
                            graph_label = np.array([[2]])
                        graph_label = np.squeeze(graph_label)
                        

                        #Adjacency
                        trk_max = int(trk_2d.shape[0]) - 1
                        adj_list = list(itertools.permutations(range(trk_max), 2))
                        adj_npy = np.array(adj_list)
                        adj_npy = adj_npy.T

                        #npy2torch summary
                        data_torch = torch.from_numpy(trk_2d)
                        label_torch = torch.from_numpy(label)
                        label_torch = label_torch.to(torch.int64)
                        adj_2d_torch = torch.from_numpy(adj_npy)
                        adj_torch = adj_2d_torch.to(torch.long)
                        graph_torch = torch.from_numpy(graph_label)
                        graph_torch = graph_torch.to(torch.long)
                
                        data = Data(x=data_torch,
                                    edge_index=adj_torch,
                                    y=graph_torch,
                                    node_y=label_torch)
                        counter += 1
                        torch.save(data, "/gluster/maxi/ilc/onoe/gnn/data_1212/processed/" + f'data_{counter}.pt')

    def len(self):
        return len(self.processed_file_names)

    def get(self, counter):
        data = torch.load("/gluster/maxi/ilc/onoe/gnn/data_1212/processed/" +f'data_{counter}.pt')
        return data
