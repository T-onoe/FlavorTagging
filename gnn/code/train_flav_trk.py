import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, global_mean_pool
import torch.optim as optimizers
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from dataset_flv_trk import TrkDataset
import matplotlib as mpl
import datetime
import time


class NodeAndGraphClassification(torch.nn.Module):
    def __init__(self):
        super(NodeAndGraphClassification, self).__init__()
        self.fc1 = nn.Linear(dataset.num_node_features, 128)
        self.sage1 = SAGEConv(128, 256)
        self.bn1 = BatchNorm(256)
        self.sage2 = SAGEConv(256, 128)
        self.bn2 = BatchNorm(128)
        self.sage3 = SAGEConv(128, 64)
        self.bn3 = BatchNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn4 = BatchNorm(64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(64,3)
        self.bn5 = BatchNorm(128)
        self.fc5 = nn.Linear(128,5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        node_out = self.fc3(x)
        node_out = self.bn5(node_out)
        node_out = F.relu(node_out)
        node_out = self.fc5(node_out)
        
        graph_out = global_mean_pool(x, data.batch)
        graph_out= self.fc4(graph_out)
        return node_out, graph_out

def combined_loss(node_output, graph_output, node_labels, graph_labels):
    #node classification loss
    node_loss = node_loss_fn(node_output, node_labels)
    #graph classification loss
    graph_loss = graph_loss_fn(graph_output, graph_labels)
    #combine
    loss = node_loss + graph_loss
    return loss
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_printoptions(precision=100)
    now = datetime.datetime.now()
    torch.cuda.synchronize()
    start = time.time()

    #Dataset prepare
    print("Loading dataset.....")
    dataset = TrkDataset(root="/gluster/maxi/ilc/onoe/gnn/data_1208/")
    print(dataset)
    num_graph = len(dataset)
    train_size = round(num_graph * 0.8)
    val_test_size = num_graph - train_size
    val_size = round(val_test_size * 0.5)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = random_split(val_dataset, [val_size, val_size])
    print("Now set dataloader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("Loading model.....")
    model = NodeAndGraphClassification().to(device)
    weights = torch.tensor([0.135, 0.050, 0.574, 0.682, 1.0]).cuda() #inverse ratio
#    weights = torch.tensor([0.3679, 0.22385, 0.75787, 0.826, 1.0]).cuda()   #Sqrt of inverse ratio
    node_loss_fn = nn.CrossEntropyLoss(weight=weights)
    graph_loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.01, amsgrad=False)
    scheduler= optimizers.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            node_out, graph_out = model(data)
            node_pred = node_out.argmax(dim=1).cpu()
            graph_pred = graph_out.argmax(dim=1).cpu()
            loss = combined_loss(node_out, graph_out, data.node_y, data.y)
            loss = loss.to(device)
            total_loss += loss.item() * data.num_graphs
            acc_node = accuracy_score(data.node_y.cpu(), node_pred)
            acc_graph = accuracy_score(data.y.cpu(), graph_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return total_loss / len(train_loader.dataset), acc_node, acc_graph

    def validation():
        model.eval()
        total_loss =0
        total_acc_node = 0
        total_acc_graph = 0
        for data in val_loader:
            data = data.to(device)
            node_out, graph_out = model(data)
            node_pred = node_out.argmax(dim=1).cpu()
            graph_pred = graph_out.argmax(dim=1).cpu()
            loss = combined_loss(node_out, graph_out, data.node_y, data.y)
            loss = loss.to(device)
            total_loss += loss.item() * data.num_graphs
            total_acc_node = accuracy_score(data.node_y.cpu(), node_pred)
            total_acc_graph = accuracy_score(torch.squeeze(data.y).cpu(), graph_pred)
        return total_loss / len(val_loader.dataset), total_acc_node, total_acc_graph

    def test():
        model.eval()
        total_loss = 0
        total_acc = 0
        total_acc_g = 0
        node_predict = []
        graph_predict= []
        target = []
        target_g = []

        pred_0 = []
        pred_1 = []
        pred_2 = []
        pred_3 = []
        pred_4 = []
        pred_0_g = []
        pred_1_g = []
        pred_2_g = []

        pred_b = [] #the probability of predicting b
        pred_c = [] #the probability of predicting c
        pred_ud = [] #the probability of predicting ud 
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                node_out, graph_out = model(data)
                #Node output preparation
                node_out_0 = node_out[:, 0]
                node_out_1 = node_out[:, 1]
                node_out_2 = node_out[:, 2]
                node_out_3 = node_out[:, 3]
                node_out_4 = node_out[:, 4]
                pred_0 += [float(l) for l in node_out_0]
                pred_1 += [float(l) for l in node_out_1]
                pred_2 += [float(l) for l in node_out_2]
                pred_3 += [float(l) for l in node_out_3]
                pred_4 += [float(l) for l in node_out_4]
                #Graph output preparation
                graph_out_0 = graph_out[:, 0]
                graph_out_1 = graph_out[:, 1]
                graph_out_2 = graph_out[:, 2]
                pred_0_g += [float(l) for l in graph_out_0]
                pred_1_g += [float(l) for l in graph_out_1]
                pred_2_g += [float(l) for l in graph_out_2]
                #For efficiency
                b_output = graph_out[:, 0]
                c_output = graph_out[:, 1]
                uds_output = graph_out[:, 2]

                pred_b += [float(l) for l in b_output]
                pred_c += [float(l) for l in c_output]
                pred_ud += [float(l) for l in uds_output]
                #Main
                node_pred = node_out.argmax(dim=1).cpu()
                graph_pred = graph_out.argmax(dim=1).cpu()
                node_pred_list = node_pred.tolist()
                node_predict.extend(node_pred_list)
                graph_pred_list = graph_pred.tolist()
                graph_predict.extend(graph_pred_list)
                target_list = data.node_y.tolist()
                target.extend(target_list)
                target_list_g = data.y.tolist()
                target_g.extend(target_list_g)
                loss = combined_loss(node_out, graph_out, data.node_y, data.y)
                loss = loss.to(device)
                total_loss += loss.item() * data.num_graphs
                total_acc = accuracy_score(data.node_y.cpu(), node_pred)
                total_acc_g = accuracy_score(data.y.cpu(), graph_pred)
        return total_loss / len(test_loader.dataset), total_acc, total_acc_g, node_predict, graph_predict, target, target_g, pred_0, pred_1, pred_2, pred_3, pred_4, pred_0_g, pred_1_g, pred_2_g, pred_b, pred_c, pred_ud

     # !!TRAIN STARTS HERE !!
    train_losses = []
    train_acc_node = []
    train_acc_graph = []
    val_losses = []
    val_accs = []
    val_accs_g = []
    print("Training Start!")
    for epoch in range(1,500):
        train_loss, acc_node, acc_graph = train()
        val_loss, val_acc, val_acc_g = validation()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc_node.append(acc_node)
        train_acc_graph.append(acc_graph)
        val_accs.append(val_acc)
        val_accs_g.append(val_acc_g)
        scheduler.step()
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} Node_Acc: {acc_node:.4f}  Graph_Acc: {acc_graph:.4f}, Val Loss: {val_loss:.4f} Node_Acc: {val_acc:.4f} Graph_Acc: {val_acc_g:.4f}')

    test_loss, test_acc, test_acc_g, node_pred, graph_pred, target, target_g, pred_0, pred_1, pred_2, pred_3, pred_4, pred_0_g, pred_1_g, pred_2_g, pred_b, pred_c, pred_ud = test()
    print(f'Total Result == Loss: {test_loss:.4f} Node_Accuracy: {test_acc:.4f} Graph_Accuracy: {test_acc_g:.4f}' )

    #report
    print("Node report")
    print(classification_report(target, node_pred))
    print("Graph report")
    print(classification_report(target_g, graph_pred))
    
    #Confusion matrix (Node)
    cm = confusion_matrix(target, node_pred)
    pre_cm_sum = cm.sum(axis=1)
    cm_sum = pre_cm_sum.reshape(5, 1)
    cm_norm = cm / cm_sum
    cm_norm = pd.DataFrame(data=cm_norm, index=["Others", "PV", "SVBB", "TVCC", "SVCC"], columns=["Others", "PV", "SVBB", "TVCC", "SVCC"])
    print(cm_norm)
    sns.heatmap(cm_norm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/cm_node_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #Confusion matrix (Graph)
    cm2 = confusion_matrix(target_g, graph_pred)
    pre_cm_sum2 = cm2.sum(axis=1)
    cm_sum2 = pre_cm_sum2.reshape(3, 1)
    cm_norm2 = cm2 / cm_sum2
    cm_norm2 = pd.DataFrame(data=cm_norm2, index=["b", "c", "ud"], columns=["b", "c", "ud"])
    print(cm_norm2)
    sns.heatmap(cm_norm2, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/cm_graph_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #ROC_curve (Node)
    fpr1, tpr1, thresholds = roc_curve(target, pred_0, pos_label=0)                               
    fpr2, tpr2, thresholds = roc_curve(target, pred_1, pos_label=1)                               
    fpr3, tpr3, thresholds = roc_curve(target, pred_2, pos_label=2)
    fpr4, tpr4, thresholds = roc_curve(target, pred_3, pos_label=3)
    fpr5, tpr5, thresholds = roc_curve(target, pred_4, pos_label=4)
    fig = plt.figure()
    plt.plot(fpr1, tpr1, marker=',', label="Others")                                            
    plt.plot(fpr2, tpr2, marker=',', label="PV")                                                   
    plt.plot(fpr3, tpr3, marker=',', label="SVBB")
    plt.plot(fpr4, tpr4, marker=',', label="TVCC")
    plt.plot(fpr5, tpr5, marker=',', label="SVBC")
    plt.xlabel('FPR: False positive rate')                                                        
    plt.ylabel('TPR: True positive rate')                                                         
    plt.grid()                                                                                    
    plt.legend()
    plt.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/roc_curve_node_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #ROC curve (Graph)
    fpr1, tpr1, thresholds = roc_curve(target_g, pred_0_g, pos_label=0)
    fpr2, tpr2, thresholds = roc_curve(target_g, pred_1_g, pos_label=1)
    fpr3, tpr3, thresholds = roc_curve(target_g, pred_2_g, pos_label=2)
    fig = plt.figure()
    plt.plot(fpr1, tpr1, marker=',', label="b")
    plt.plot(fpr2, tpr2, marker=',', label="c")
    plt.plot(fpr3, tpr3, marker=',', label="ud")
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.legend()
    plt.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/roc_curve_graph_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #Loss
    fig = plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='valid_loss')
    plt.yscale('log')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.legend(loc='best')
    fig.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/loss_' + now.strftime('%Y%m%d_%H%M') + '.png')
    #Accuracy (Node)
    fig = plt.figure()
    plt.plot(train_acc_node, label='train_acc')
    plt.plot(val_accs, label='valid_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    fig.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/acc_node_' + now.strftime('%Y%m%d_%H%M') + '.png')
    #Accuracy (Graph)
    fig = plt.figure()
    plt.plot(train_acc_graph, label='train_acc')
    plt.plot(val_accs_g, label='valid_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    fig.savefig('/home/onoe/ILC/Deep_Learning/gnn/output/planb/acc_graph_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #Efficiency
    bb_eff = []
    bc_back = []
    buds_back = []
    cb_back = []
    cc_eff = []
    cuds_back = []
    threshold = [i /10 for i in range(0, 10, 1)]
    threshold.extend([1.0])
    threshold.extend([i /100 for i in range(1, 10, 5)])
    threshold.extend([i /1000 for i in range(1, 10, 5)])
    threshold.extend([i /10000 for i in range(1, 10, 5)])
    threshold.extend([i /100000 for i in range(1, 10, 5)])
    threshold.extend([i /1000000 for i in range(1, 10, 5)])
    threshold.extend([0.9 + i /100 for i in range(1, 10, 5)])
    threshold.extend([0.99 + i /1000 for i in range(1, 10, 5)])
    threshold.extend([0.999 + i /10000 for i in range(1, 10, 5)])
    threshold.extend([0.999 + i /100000 for i in range(1, 10, 5)])
    threshold.extend([0.999 + i /1000000 for i in range(1, 10, 5)])
    for i in threshold:
        #(target)_(pred)
        b_b = []
        b_c = []
        b_uds = []
        c_b = []
        c_c = []
        c_uds = []
        #正しい答えの数を得るためのリスト
        num_b = []
        num_c = []
        num_uds = []

        for j in range(len(target_g)):
            if target_g[j] == 0:
              num_b.append(target_g[j])
              if pred_b[j] >= i:
                b_b.append(pred_b[j])
              if pred_c[j] >= i:
                b_c.append(pred_c[j])
              if pred_ud[j] >= i:
                b_uds.append(pred_ud[j])
            elif target_g[j] == 1:
              num_c.append(target_g[j])
              if pred_b[j] >= i:
                c_b.append(pred_b[j])
              if pred_c[j] >= i:
                c_c.append(pred_c[j])
              if pred_ud[j] >= i:
                c_uds.append(pred_ud[j])
            elif target_g[j] == 2:
              num_uds.append(target_g[j])

        #number of data
        num_b_b = len(b_b)
        num_b_c = len(b_c)
        num_b_uds = len(b_uds)
        num_c_b = len(c_b)
        num_c_c = len(c_c)
        num_c_uds = len(c_uds)
        len_b = len(num_b)
        len_c = len(num_c)
        len_uds = len(num_uds)
        #calculate the ratio
        rate_b_b = num_b_b / len_b
        rate_b_c = num_b_c / len_b
        rate_b_uds = num_b_uds / len_b
        rate_c_b = num_c_b / len_c
        rate_c_c = num_c_c / len_c
        rate_c_uds = num_c_uds / len_c
        #add every threshold
        bb_eff.append(rate_b_b)
        bc_back.append(rate_b_c)
        buds_back.append(rate_b_uds)
        cb_back.append(rate_c_b)
        cc_eff.append(rate_c_c)
        cuds_back.append(rate_c_uds)
        
    fig = plt.figure()
    plt.scatter(bb_eff, bc_back, s=8,  facecolors="None", linewidths=0.8, edgecolors='lime' ,label='c jets')
    plt.scatter(bb_eff, buds_back, s=8, facecolors="None", linewidths=0.8, edgecolors='blue', label='uds jets')
    plt.title('b tag with background')
    plt.grid(axis='both', linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.ylim(10**(-4), 1)
    plt.xlabel('Tagging efficiency')
    plt.ylabel('Mis-id. fraction')
    plt.legend()
    fig.savefig('../../output/planb/eff_b_' + now.strftime('%Y%m%d_%H%M') + '.png')

    fig = plt.figure()
    plt.scatter(cc_eff, cb_back, s=8,facecolors="None", linewidths=0.8, edgecolors='red', label='b jets')
    plt.scatter(cc_eff, cuds_back, s=8,facecolors="None", linewidths=0.8, edgecolors='black', label='uds jets')
    plt.title('c tag with background')
    plt.grid(axis='both', linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.ylim(10**(-4), 1)
    plt.xlabel('Tagging efficiency')
    plt.ylabel('Mis-id. fraction')
    plt.legend()
    fig.savefig('../../output/planb/eff_c_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #End
    print('saving as '+ now.strftime('%Y%m%d_%H%M'))
    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    #Convert second to hour, minute and seconds                                                   
    elapsed_hour = int(elapsed_time // 3600)
    elapsed_minute = int((elapsed_time % 3600) // 60)
    elapsed_second = int(elapsed_time % 3600 % 60)

    print(str(elapsed_hour).zfill(2) + ":" + str(elapsed_minute).zfill(2) + ":" + str(elapsed_second).zfill(2))
    print(elapsed_time, 'sec.')
