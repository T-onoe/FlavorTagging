import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizers
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as transforms
from sklearn.model_selection import train_test_split
#from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import pandas as pd
#import seaborn as sns
import datetime
import time
import matplotlib as mpl


def my_softmax(x):
    x = x.to('cpu' , dtype=torch.double)
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    out = x_exp/x_exp_sum
    sum = torch.sum(out,1,dtype=torch.double)
    sum = sum.unsqueeze(1)
    return torch.div(out,sum)
    
class TaggingNet(nn.Module):
    def __init__(self):
        super(TaggingNet, self).__init__()
        self.fc1 = nn.Linear(42, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dp1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.dp2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(100, 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.dp3 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(100, 3)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dp1(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dp2(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.dp3(x)
        x = self.LeakyReLU(x)
        x = self.fc5(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_printoptions(precision=100)
    now = datetime.datetime.now()
    torch.cuda.synchronize()
    start = time.time()
    mpl.rcParams['agg.path.chunksize'] = 1000000

    #data
    x1 = np.load('/home/onoe/ILC/Deep_Learning/dnn/data/npy/1mpure/categories/v01/v01-05/Pn23n23h_bb_sum.npy')
    x2 = np.load('/home/onoe/ILC/Deep_Learning/dnn/data/npy/1mpure/categories/v01/v01-05/Pn23n23h_cc_sum.npy')
#    x3 = np.load('/home/onoe/ILC/Deep_Learning/dnn/data/npy/1mpure/categories/v01/v01-05/Pn23n23h_gg_sum.npy')
    x4 = np.load('/home/onoe/ILC/Deep_Learning/dnn/data/npy/1mpure/categories/v01/v01-05/Pn23n23h_qq_sum.npy')
    #number of events
    bb = x1.shape[0]
    cc = x2.shape[0]
#    gg = x3.shape[0]
    qq = x4.shape[0]
    #label                                                                                                                     
    y1 = np.full((bb), 0, dtype=np.long)
    y2 = np.full((cc), 1, dtype=np.long)
#    y3 = np.full((gg), 2, dtype=np.long)
    y4 = np.full((qq), 2, dtype=np.long)
    #merge
    x = np.concatenate([x1,x2,x4],0)
    y = np.concatenate([y1,y2,y4],0)
    #split to train and val
    x_trains, x_test, y_trains, y_test = train_test_split(x, y,test_size=0.2,random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_trains, y_trains,test_size=0.25,random_state=1)
    #toTensor
    data_train = torch.tensor(x_train, dtype=torch.double)
    label_train = torch.tensor(y_train, dtype=torch.double)
    data_val = torch.tensor(x_val, dtype=torch.double)
    label_val = torch.tensor(y_val, dtype=torch.double)
    data_test = torch.tensor(x_test, dtype=torch.double)
    label_test = torch.tensor(y_test, dtype=torch.double)
    #DataLoader
    dataset_train = TensorDataset(data_train, label_train)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_test = TensorDataset(data_test, label_test)
    train_dataloader = DataLoader(dataset_train, shuffle=False, batch_size=1024)
    val_dataloader = DataLoader(dataset_val, shuffle=False, batch_size=1024)
    test_dataloader = DataLoader(dataset_test, shuffle=False, batch_size=1024)

    # model
    model = TaggingNet().to(device)

    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.01, amsgrad=False)
    scheduler= optimizers.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
#    optimizer = optimizers.SGD(model.parameters(), lr=1e-10, momentum=0.9)

    def compute_loss(t, y):
        return criterion(y, t)

    def train_step(x, t):
        model.train()
        x_t = model(x)
        output = f.softmax(x_t, dim=1)
        loss = compute_loss(t, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, output

    def val_step(x, t):
        model.eval()
        output = model(x)
        x_t = model(x)
        output = f.softmax(x_t, dim=1)
        loss = criterion(output, t)
        return loss, output

    def test_step(x, t):
        model.eval()
        with torch.no_grad():
            x_t = model(x)
            output = my_softmax(x_t)
            output_gpu = output.to('cuda')
            loss = criterion(output_gpu, t)
            return loss, output

    epochs = 2
    hist = {'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):        
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.

        for (x, t) in train_dataloader:
            x, t = x.to(device=device, dtype=torch.float), t.to(device=device, dtype=torch.long)
            print(t.size())
            loss, output = train_step(x, t)
            print(output.size())
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t.tolist(),
                               output.argmax(dim=-1).tolist())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        for (x, t) in val_dataloader:
            x, t = x.to(device=device, dtype=torch.float), t.to(device=device, dtype=torch.long)
            loss, output = val_step(x, t)
            val_loss += loss.item()
            val_acc += \
                accuracy_score(t.tolist(),
                               output.argmax(dim=-1).tolist())
            
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        hist['loss'].append(train_loss)
        hist['accuracy'].append(train_acc)
        hist['val_loss'].append(val_loss)
        hist['val_accuracy'].append(val_acc)

        print('epoch: {}, loss: {:.3}, acc: {:.3f}'
              ', val_loss: {:.3}, val_acc: {:.3f}'.format(
                  epoch+1,
                  train_loss,
                  train_acc,
                  val_loss,
                  val_acc
              ))

    #test
    test_loss = 0.
    test_acc = 0.
    output_all = np.empty((0,3), np.float64)
    pred = []
    pred_b = [] #the probability of predicting b
    pred_c = [] #the probability of predicting c
    pred_uds = [] #the probability of predicting uds
    target = []
    
    for (x, t) in test_dataloader:
        x, t = x.to(device=device, dtype=torch.float), t.to(device=device, dtype=torch.long)
        loss, output = test_step(x, t)
        test_loss += loss.item()
        test_acc += \
            accuracy_score(t.tolist(),
                           output.argmax(dim=-1).tolist())
        output_npy = output.to('cpu').detach().numpy().copy()
        output_all = np.append(output_all, output_npy, axis=0)
        b_output = output[:, 0]
        c_output = output[:, 1]
        uds_output = output[:, 2]

        pred_b += [float(l) for l in b_output]
        pred_c += [float(l) for l in c_output]
        pred_uds += [float(l) for l in uds_output]
        pred += [float(l.argmax()) for l in output]
        target += [int(l) for l in t]

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

#    np.save("/home/onoe/ILC/Deep_Learning/dnn/macro/out/output/output_" + now.strftime('%Y%m%d_%H%M') + ".npy", output_all, fix_imports=True)
    
    print('Test loss: {:.3}, Test acc: {:.3f}'.format(
              test_loss,
              test_acc
          ))

#report
    print(classification_report(target, pred))

#Confusion matrix
    cm = confusion_matrix(target, pred)
    pre_cm_sum = cm.sum(axis=1)
    cm_sum = pre_cm_sum.reshape(3, 1)
    cm_norm = cm / cm_sum
    cm_norm = pd.DataFrame(data=cm_norm, index=["b", "c", "uds"],
                           columns=["b", "c", "uds"])
    print(cm_norm)
    sns.heatmap(cm_norm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig('/home/onoe/ILC/Deep_Learning/dnn/macro/out/cm/confusion_matrix_' + now.strftime('%Y%m%d_%H%M') + '.png')
    """
#roc curve
    # b curve
    fpr1, tpr1, thresholds = roc_curve(target, pred_b, pos_label=0)
    fpr2, tpr2, thresholds = roc_curve(target, pred_c, pos_label=1)
    fpr3, tpr3, thresholds = roc_curve(target, pred_uds, pos_label=2)
    fig = plt.figure()
    plt.plot(fpr1, tpr1, marker=',', label="b")
    plt.plot(fpr2, tpr2, marker=',', label="c")
    plt.plot(fpr3, tpr3, marker=',', label="uds")
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.legend()
#    plt.savefig('/home/onoe/ILC/Deep_Learning/dnn/macro/out/plot/roc_curve_' + now.strftime('%Y%m%d_%H%M') + '.png')
    """
#efficiency curve
    #list for plot
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
    threshold.extend([i /10000000 for i in range(1, 10, 5)])
    threshold.extend([i /100000000 for i in range(1, 10, 5)])
    threshold.extend([i /1000000000 for i in range(1, 10, 5)])
    threshold.extend([i /10000000000 for i in range(1, 10, 5)])
    threshold.extend([i /100000000000 for i in range(1, 10, 5)])
    threshold.extend([i /1000000000000 for i in range(1, 10, 5)])
    threshold.extend([i /10000000000000 for i in range(1, 10, 5)])
    threshold.extend([i /100000000000000 for i in range(1, 10, 5)])
    threshold.extend([i /1000000000000000 for i in range(1, 10, 5)])
    threshold.extend([0.9 + i /100 for i in range(1, 10, 5)])
    threshold.extend([0.99 + i /1000 for i in range(1, 10, 5)])
    threshold.extend([0.999 + i /10000 for i in range(1, 10, 5)])
    threshold.extend([0.9999 + i /100000 for i in range(1, 10, 5)])
    threshold.extend([0.99999 + i /1000000 for i in range(1, 10, 5)])
    threshold.extend([0.999999 + i /10000000 for i in range(1, 10, 5)])
    threshold.extend([0.9999999 + i /100000000 for i in range(1, 10, 5)])
    threshold.extend([0.99999999 + i /1000000000 for i in range(1, 10, 5)])
    threshold.extend([0.999999999 + i /10000000000 for i in range(1, 10, 5)])
    threshold.extend([0.9999999999 + i /100000000000 for i in range(1, 10, 5)])
    threshold.extend([0.99999999999 + i /1000000000000 for i in range(1, 10, 5)])
    threshold.extend([0.999999999999 + i /10000000000000 for i in range(1, 10, 5)])
    threshold.extend([0.9999999999999 + i /10000000000000 for i in range(1, 10, 5)])
    threshold.extend([0.99999999999999 + i /100000000000000 for i in range(1, 10, 5)])
    
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
        
        for j in range(len(target)):
            if target[j] == 0:
              num_b.append(target[j])
              if pred_b[j] >= i:
                b_b.append(pred_b[j])
              if pred_c[j] >= i:
                b_c.append(pred_c[j])
              if pred_uds[j] >= i:
                b_uds.append(pred_uds[j])
            elif target[j] == 1:
              num_c.append(target[j])
              if pred_b[j] >= i:
                c_b.append(pred_b[j])
              if pred_c[j] >= i:
                c_c.append(pred_c[j])
              if pred_uds[j] >= i:
                c_uds.append(pred_uds[j])
            elif target[j] == 2:
              num_uds.append(target[j])
              
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
    plt.xlabel('Tagging efficiency')
    plt.ylabel('Mis-id. fraction')
    plt.legend()
    fig.savefig('../out/plot/eff_b_' + now.strftime('%Y%m%d_%H%M') + '.png') 

    fig = plt.figure()
    plt.scatter(cc_eff, cb_back, s=8,facecolors="None", linewidths=0.8, edgecolors='red', label='b jets')
    plt.scatter(cc_eff, cuds_back, s=8,facecolors="None", linewidths=0.8, edgecolors='black', label='uds jets')
    plt.title('c tag with background')
    plt.grid(axis='both', linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.xlabel('Tagging efficiency')
    plt.ylabel('Mis-id. fraction')
    plt.legend()
    fig.savefig('../out/plot/eff_c_' + now.strftime('%Y%m%d_%H%M') + '.png')
    
    #Save Weight
#    torch.save(model.state_dict(), '/home/onoe/ILC/Deep_Learning/dnn/macro/out/weight/model_weight_3_' + now.strftime('%Y%m%d_%H%M') + '.pth')
    
    #Plot output
    loss = hist['loss']
    val_loss = hist['val_loss']
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']

    #plot loss
    fig = plt.figure()
    plt.plot(range(len(loss)), loss,
             linewidth=1,
             label='loss')
    plt.plot(range(len(val_loss)), val_loss,
             linewidth=1,
             label='val_loss')
    plt.title('model loss')
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='best')
#    plt.show()
    fig.savefig('../out/plot/loss_' + now.strftime('%Y%m%d_%H%M') + '.png')

    #plot acc
    fig = plt.figure()
    plt.plot(range(len(acc)), acc,
             linewidth=1,
             label='acc')
    plt.plot(range(len(val_acc)), val_acc,
             linewidth=1,
             label='val_acc')
    plt.title('model accuracy')
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
#    plt.show()
    fig.savefig('../out/plot/acc_' + now.strftime('%Y%m%d_%H%M') + '.png')

#    End 
    print('saving as '+ now.strftime('%Y%m%d_%H%M'))
    torch.cuda.synchronize()
    elapsed_time = time.time() - start
#    convert second to hour, minute and seconds
    elapsed_hour = int(elapsed_time // 3600)
    elapsed_minute = int((elapsed_time % 3600) // 60)
    elapsed_second = int(elapsed_time % 3600 % 60)

    print(str(elapsed_hour).zfill(2) + ":" + str(elapsed_minute).zfill(2) + ":" + str(elapsed_second).zfill(2))

    print(elapsed_time, 'sec.')
