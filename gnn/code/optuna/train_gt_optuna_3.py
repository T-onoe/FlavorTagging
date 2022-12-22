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
from dataset_graph_node import TrkDataset
import matplotlib as mpl
import datetime
import time
import optuna

class NodeAndGraphClassification(torch.nn.Module):
    def __init__(self, fc2, fc3, hidden_graph):
        super(NodeAndGraphClassification, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.sage1 = SAGEConv(128, hidden_graph)
        self.bn1 = BatchNorm(hidden_graph)
        self.sage2 = SAGEConv(hidden_graph, hidden_graph)
        self.bn2 = BatchNorm(hidden_graph)
        self.sage3 = SAGEConv(hidden_graph, hidden_graph)
        self.bn3 = BatchNorm(hidden_graph)
        self.fc2 = nn.Linear(hidden_graph, fc2)
        self.bn4 = BatchNorm(fc2)
        self.fc3 = nn.Linear(fc2,fc3)
        self.fc4 = nn.Linear(fc2,3)
        self.bn5 = BatchNorm(fc3)
        self.fc5 = nn.Linear(fc3,5)

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

def combined_loss(alpha, node_output, graph_output, node_labels, graph_labels):
    weights = torch.tensor([0.135, 0.050, 0.574, 0.682, 1.0]).cuda()
    node_loss_fn = nn.CrossEntropyLoss(weight=weights)
    graph_loss_fn = nn.CrossEntropyLoss()
    #node classification loss
    node_loss = node_loss_fn(node_output, node_labels)
    #graph classification loss
    graph_loss = graph_loss_fn(graph_output, graph_labels)
    #combine
    loss = alpha * node_loss + graph_loss
    return loss, graph_loss

def objective(trial):
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    #Optimize variables
#    fc1 = trial.suggest_int('fc1', 16, 128)
    fc2 = trial.suggest_int("fc2", 16, 1024)
    fc3 = trial.suggest_int("fc3", 16, 1024)
    hidden_graph = trial.suggest_int("hidden_graph", 16, 1024)
    lr = trial.suggest_float("lr", 0.00001, 0.1)
    alpha = trial.suggest_float("alpha", 0.5, 2.0)
#    num_epochs = trial.suggest_int("num_epochs", 10, 100)

    model = NodeAndGraphClassification(fc2, fc3, hidden_graph).to(device)
    optimizer = optimizers.Adam(model.parameters(), lr=lr, amsgrad=False)
    weights = torch.tensor([0.135, 0.050, 0.574, 0.682, 1.0]).cuda()
    node_loss_fn = nn.CrossEntropyLoss(weight=weights)
    graph_loss_fn = nn.CrossEntropyLoss()
#    val_loss = 0

    def train():
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            node_out, graph_out = model(data)
            loss, graph_loss = combined_loss(alpha, node_out, graph_out, data.node_y, data.y)
            loss = loss.to(device)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    def validation():
        model.eval()
        total_loss =0
        for data in val_loader:
            data = data.to(device)
            node_out, graph_out = model(data)
            node_pred = node_out.argmax(dim=1).cpu()
            graph_pred = graph_out.argmax(dim=1).cpu()
            loss, graph_loss = combined_loss(alpha, node_out, graph_out, data.node_y, data.y)
            graph_loss = graph_loss.to(device)
            total_loss += graph_loss.item() * data.num_graphs
        return total_loss / len(val_loader.dataset)

    
    for epoch in range(100):
        train_loss = train()
    val_graph_loss = validation()

    print("Running trial with params: " + str(trial.params))
    return val_graph_loss

if __name__=='__main__':
    study = optuna.create_study(storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                                    study_name="flavortagging_1k")
    study.optimize(objective, n_trials=10)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
    print("results")
    print(study.best_trial)
