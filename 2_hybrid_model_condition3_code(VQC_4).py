import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from sklearn.utils import class_weight
from collections import Counter
from os import listdir
import csv

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader

import pennylane as qml
from pennylane import numpy as anp


# Unitary Ansatze for Convolutional Layer
def U_TTN(params, wires):  # 2 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_5(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_6(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRX(params[4], wires=[wires[1], wires[0]])
    qml.CRX(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_9(params, wires):  # 2 params
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])


def U_13(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRZ(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])


def U_14(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRX(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRX(params[5], wires=[wires[0], wires[1]])


def U_15(params, wires):  # 4 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_SO4(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

# Pooling Layer
def Pooling_ansatz1(params, wires): #2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])

def Pooling_ansatz2(wires): #0 params
    qml.CRZ(wires=[wires[0], wires[1]])

def Pooling_ansatz3(*params, wires): #3 params
    qml.CRot(*params, wires=[wires[0], wires[1]])

def conv_layer1(U, params):
    U(params, wires=[0, 3])
    U(params, wires=[0, 1])
    U(params, wires=[2, 3])
    U(params, wires=[1, 2])
def conv_layer2(U, params):
    U(params, wires=[0, 2])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    V(params, wires=[1, 0])
    V(params, wires=[3, 2])
def pooling_layer2(V, params):
    V(params, wires=[2,0])

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, params):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    #qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Z')
    
    U_params = 15
    total_params = U_params * 2 + 2 * 2
    #params = anp.random.randn(total_params, requires_grad=True)
    params = torch.randn(total_params, requires_grad=True)
    
    param1 = params[0: U_params]
    param2 = params[U_params: 2 * U_params]    
    param3 = params[2 * U_params: 2 * U_params + 2]
    param4 = params[2 * U_params + 2: 2 * U_params + 4]

    # Pooling Ansatz1 is used by default
    conv_layer1(U_SU4, param1)
    pooling_layer1(Pooling_ansatz1, param3)
    conv_layer2(U_SU4, param2)
    pooling_layer2(Pooling_ansatz1, param4)
    
    result = qml.probs(wires=0)
    
    return result


ansatz = 'U_SU4'
n_layers = 4
weight_shapes = {"params": (34, 1)}
qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)


# ### Import rs-fMRI (as a image)

# #### Define dataloader
class fMRIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset_ = dataset
        print(self.dataset_.head())
        #print(type(self.dataset_))
        self.dataset_df = pd.DataFrame(self.dataset_)
        self.dataset = torch.from_numpy(self.dataset_df.iloc[:,:].values)

    def __len__(self):
        return (self.dataset.shape[0])

    def __getitem__(self, idx):
        return self.dataset[:,:-1][idx].float(), self.dataset[:,-1][idx].float()


# #### Define Hybrid(quatum-classical) model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 30, 2)
        self.conv2 = nn.Conv1d(8, 16, 20, 2)
        self.dropout = nn.Dropout1d()
        self.fc1 = nn.Linear(56, 18)
        self.fc2 = nn.Linear(18, 9)
        self.fc3 = nn.Linear(9, 2)

    def forward(self, x):
        x_splitted_0 = x[0][:16].view(1,1,16)
        x_splitted_1 = x[0][16:32].view(1,1,16)
        x_splitted_2 = x[0][32:48].view(1, 1, 16)
        x_splitted_3 = x[0][48:64].view(1, 1, 16)

        x_splitted_4 = x[0][64:].view(1,1,76)
        
        ## 1) classical conv layer
        x_classic = torch.tanh(self.conv1(x_splitted_4))
        x_classic = torch.tanh(self.conv2(x_classic))
        x_classic = self.dropout(x_classic)
        x_classic = x_classic.view(-1)
        
        ## 2) quantum conv layer
        x_1 = qlayer1(x_splitted_0).view(-1)
        x_2 = qlayer2(x_splitted_1).view(-1)
        x_3 = qlayer3(x_splitted_2).view(-1)
        x_4 = qlayer4(x_splitted_3).view(-1)


        concatted_x = torch.cat([x_1, x_2, x_3, x_4, x_classic]).view(1, 56)
        final_x = torch.tanh(self.fc1(concatted_x))
        final_x = torch.tanh(self.fc2(final_x))
        final_x_ = torch.softmax(self.fc3(final_x), dim=1)

        return final_x_

root_dir = "F://Student/CJG/220916_QC/0_data/1_preprocessed/1_results_221115_CN_EMCI/"


# #### Define datafile check function
def file_check(signal_df):
    class1_df = signal_df.loc[signal_df.iloc[:,-1] == 0]
    class2_df = signal_df.loc[signal_df.iloc[:,-1] == 1]
    print("Class1(patient):",class1_df.shape, "/ Class2(normal):",class2_df.shape)
    
    signal_df_fixed = pd.concat([class1_df, class2_df], axis=0)
    print("Shape of checked signal:", signal_df_fixed.shape)

    signal_data = signal_df_fixed.iloc[:,:-1]
    signal_label = signal_df_fixed.iloc[:,-1]
    print("Data:",signal_data.shape,"/Label:",signal_label.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(signal_data, signal_label, stratify=signal_label, test_size=0.2)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
    test_df_shuffled = test_df.sample(frac=1).reset_index(drop=True)
    print("Train:", train_df_shuffled.shape,"/ Test:", test_df_shuffled.shape)
    return train_df, test_df

def file_check_fixed(signal_df):
    class1_df = signal_df.loc[signal_df.iloc[:,-1] == 0]
    class2_df = signal_df.loc[signal_df.iloc[:,-1] == 1]
    print("Class1(patient) / label 0 :",class1_df.shape, "/ Class2(normal):",class2_df.shape)
    signal_df_fixed = pd.concat([class1_df, class2_df], axis=0)
    print("Shape of checked signal:", signal_df_fixed.shape)

    signal_data = signal_df_fixed.iloc[:,:-1]
    signal_label = signal_df_fixed.iloc[:,-1]
    print("Data:",signal_data.shape,"/Label:",signal_label.shape)

    return signal_data, signal_label


# #### Define figure save function
def save_fig(loss_list, mode, cv_no):
    plt.plot(loss_list)
    plt.title('Hybrid NN Training Convergence')
    plt.xlabel('Training Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig('F:/Student/CJG/220916_QC/3_IBM_QLAB/2_result/0_hybrid/0_figure/2_COND2_VQC5_front/Test_Figure_loss_'+str(mode)+'_'+str(cv_no)+'CV_'+str(ansatz)+'_.png')


# #### Define evaluation function
def evaluation(label_pred_list):
    label_pred_list_fixed = [[torch.FloatTensor([np.argmin(a[0])]), torch.FloatTensor([np.argmin(a[1])])] for a in label_pred_list]
    #label_pred_list_fixed = [[torch.FloatTensor([np.argmin(a[0])]), a[1]] for a in label_pred_list]
    #print("Fixed!:",label_pred_list_fixed[0])
    #print("Check!:", label_pred_list[0])
    TN = 0 + 0.0000000000001
    FN = 0 + 0.0000000000001
    FP = 0 + 0.0000000000001
    TP = 0 + 0.0000000000001
    for ziped in label_pred_list_fixed:
        l = float(ziped[1])
        p = float(ziped[0])
        #print("label:",l,"/ pred:",p)
        if float(l) == 1.0 and float(p) == 1.0:
            TP = TP + 1
        elif float(l) == 1.0 and float(p) == 0.0:
            FN = FN + 1
        elif float(l) == 0.0 and float(p) == 1.0:
            FP = FP + 1
        elif float(l) == 0.0 and float(p) == 0.0:
            TN = TN + 1
        
    return TN, FN, FP, TP


# #### Define training and evaluation function
def train(epochs, train_loader, test_loader, model, loss_func_train):
    loss_list = []
    inter_test_list= []
    model.train()
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data_fixed_2d = data#.view(1,1,140)

            optimizer.zero_grad()
            # Forward pass
            output = model(data_fixed_2d)[0]
            #print("Prediction:", output, output.size())
            #print("Target:", target, target.size())
            
            if target == torch.tensor([0.]):
                target_fixed = torch.tensor([1., 0.]) 
            if target == torch.tensor([1.]):
                target_fixed = torch.tensor([0., 1.])
            
            # Calculating loss
            loss = loss_func_train(output, target_fixed)
            
            #print("loss:",loss)
            #print(datetime.datetime.now())
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()

            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
            100. * (epoch + 1) / epochs, loss_list[-1]))
        
        intermediate_result = test(test_loader=test_loader, mode='test', model=model, loss_func_test=loss_func_train)
        inter_test_list.append(intermediate_result[1])
        
    return loss_list, model, inter_test_list

def test(test_loader, mode, model, loss_func_test):
    total_loss=[]
    model.eval()
    with torch.no_grad():

        pred_target=[]
        for batch_idx, (data, target) in enumerate(test_loader):
            data_fixed_2d = data#.view(1,1,140)
            output = model(data_fixed_2d)[0]
            
            if target == torch.tensor([0.]):
                target_fixed = torch.tensor([1., 0.]) 
            if target == torch.tensor([1.]):
                target_fixed = torch.tensor([0., 1.])
            
            #pred = output.argmax(dim=1, keepdim=True) 
            #print("Prediction in test:", output)
            #print("Target:", target)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            pred_target.append([output, target_fixed])
            
            loss = loss_func_test(output, target_fixed)
            total_loss.append(loss.item())

        Loss_ = sum(total_loss) / len(total_loss)
        #Accuracy_ = correct / len(test_loader) * 100
        TN, FN, FP, TP = evaluation(pred_target)
        print("TN:",TN,"/FN:",FN,"/FP:",FP,"/TP:",TP)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1_score = 2 * ((recall*precision) / (recall+precision))
        balanced_accuracy = (1/2) * ((TP / (TP+FN))+(TN / (TN+FP)))
        print("Precision:",precision,"/Recall:",recall,"/F1-score:",f1_score,"/Balanced accuracy:",balanced_accuracy)
        total_loss=[]
        
    return Loss_, [TN, FN, FP, TP, precision, recall, f1_score, balanced_accuracy]

epochs = 100

filelist = listdir(root_dir)
print("No. of files:", len(filelist))


filelist_fixed = ['[CN_EMCI]ROI_25_df_length_140.csv','[CN_EMCI]ROI_26_df_length_140.csv',
                  '[CN_EMCI]ROI_27_df_length_140.csv','[CN_EMCI]ROI_28_df_length_140.csv',
                  '[CN_EMCI]ROI_29_df_length_140.csv','[CN_EMCI]ROI_30_df_length_140.csv',
                  '[CN_EMCI]ROI_31_df_length_140.csv','[CN_EMCI]ROI_32_df_length_140.csv',
                  '[CN_EMCI]ROI_33_df_length_140.csv','[CN_EMCI]ROI_34_df_length_140.csv',
                  '[CN_EMCI]ROI_35_df_length_140.csv','[CN_EMCI]ROI_36_df_length_140.csv',
                  '[CN_EMCI]ROI_37_df_length_140.csv']


def read_csv(file_dir, file):
    rows = []
    with open(file_dir+"/"+str(file), 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    return pd.DataFrame(rows).astype('float64') 

def normalize(x, newRange=(0, 1)): #x is an array. Default range is between zero and one
    xmin, xmax = np.min(x), np.max(x) #get max and min from input array
    norm = (x - xmin)/(xmax - xmin) # scale between zero and one
    
    if newRange == (0, 1):
        return(norm) # wanted range is the same as norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]


for no, file in enumerate(filelist_fixed[:]):
    print("No:",str(no),"/File:",str(file))
    ROI_info = str(file).split('_')[2]

    
    datafile = read_csv(file_dir=root_dir, file=str(file))
    #datafile = pd.read_csv(root_dir+"/"+str(file), header=None, skip_blank_lines=True)
    #datafile = np.loadtxt(root_dir+"/"+str(file), delimiter=',')
    
    print("Shape of loaded File:", datafile.shape)
    signal_data, signal_label = file_check_fixed(datafile)
    
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(signal_data, signal_label)
    
    for no_, (train_index, test_index) in enumerate(skf.split(signal_data, signal_label)):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = signal_data.iloc[train_index], signal_data.iloc[test_index]
        y_train, y_test = signal_label.iloc[train_index], signal_label.iloc[test_index]
        print("-------------------------------------")
        print(str(no_+1)+"-Fold")
        print("X train:", X_train.shape, "/ y train:", y_train.shape,"/ X test:", X_test.shape,"/ y test:",y_test.shape)
        print("Class labels in y train:", Counter(y_train.values.tolist()))
        print("Class labels in y test:", Counter(y_test.values.tolist()))
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        print("Train:", train_df.shape,"/Test:", test_df.shape)

    
        hybrid_model = Net()
        optimizer = optim.Adam(hybrid_model.parameters(), lr=0.0001)
        print("check:", np.unique(datafile.iloc[:,-1]))
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(datafile.iloc[:,-1]),
                                                          y=datafile.iloc[:,-1])

        #class_weights_fixed = [class_weights[1], class_weights[0]]
        print("Class weight:",class_weights)
        #loss_function = nn.NLLLoss(weight=torch.FloatTensor(class_weights))
        loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
        
        tr_dataset = fMRIDataset(train_df)
        te_dataset = fMRIDataset(test_df)
        train_loader = DataLoader(tr_dataset,batch_size=1,shuffle=True,drop_last=True)
        test_loader = DataLoader(te_dataset,batch_size=1,shuffle=False,drop_last=False)
    
        loss_list_train, trained_hybrid_model, inter_list_train = train(train_loader=train_loader, test_loader=test_loader , epochs=epochs, model=hybrid_model, loss_func_train=loss_function)
        
        inter_list_train_df = pd.DataFrame(inter_list_train)
        inter_list_train_df.to_csv('F:/Student/CJG/220916_QC/3_IBM_QLAB/2_result/0_hybrid/1_loss/2_COND2_VQC5_front/Test_per_epoch(ROI_'+str(ROI_info)+')_'+str(no_)+'CV_'+str(ansatz)+'.csv', index=False, header=None)
    
        save_dir = 'F://Student/CJG/220916_QC/3_IBM_QLAB/2_result/0_hybrid/1_trained_model/2_COND2_VQC5_front/Trained_model_ROI_'+str(ROI_info)+'_'+str(no_)+'CV_'+str(ansatz)+'.pth'
        torch.save(trained_hybrid_model.state_dict(), save_dir)
        
        train_loss_df = pd.DataFrame(loss_list_train)
        train_loss_df.to_csv('F:/Student/CJG/220916_QC/3_IBM_QLAB/2_result/0_hybrid/1_loss/2_COND2_VQC5_front/Train_loss(ROI_'+str(ROI_info)+')_'+str(no_)+'CV_'+str(ansatz)+'.csv', index=False, header=None)
        save_fig(loss_list_train, mode='train_ROI_'+str(ROI_info)+'_'+str(no), cv_no=str(no_))
        loss_list_train=[]
    
        train_loss, train_perf_index = test(train_loader, mode='train',model=hybrid_model, loss_func_test=loss_function)
        test_loss, test_perf_index = test(test_loader, mode='train',model=hybrid_model, loss_func_test=loss_function)
        print("[Result] ROI_"+str(ROI_info)+": [Train] loss:"+str(train_loss)+"/ Index:"+str(train_perf_index)+
              "[Test] loss:"+str(test_loss)+"/ Index:"+str(test_perf_index))

        exp_results=[str(file), str(ROI_info), str(no_),
                          'Train',str(train_loss),str(train_perf_index[0]),str(train_perf_index[1]),str(train_perf_index[2]),str(train_perf_index[3]),
                            str(train_perf_index[4]),str(train_perf_index[5]),str(train_perf_index[6]),str(train_perf_index[7]),
                           'Test',str(test_loss),str(test_perf_index[0]),str(test_perf_index[1]),str(test_perf_index[2]),str(test_perf_index[3]),
                            str(test_perf_index[4]),str(test_perf_index[5]),str(test_perf_index[6]),str(test_perf_index[7])]

        exp_results_df = pd.DataFrame(exp_results).T
        exp_results_df.columns = ['File','ROI', 'CV',
                                    'Mode1','loss','TN','FN','FP','TP', 'precision', 'recall', 'f1_score', 'balanced_accuracy',
                                    'Mode2','loss','TN','FN','FP','TP', 'precision', 'recall', 'f1_score', 'balanced_accuracy']

        print(exp_results_df.shape)
        exp_results_df.to_csv('F:/Student/CJG/220916_QC/3_IBM_QLAB/2_result/0_hybrid/2_performance/1_COND1_VQC1_front/performance_ROI_'+str(ROI_info)+'_'+str(no_)+'CV_'+str(ansatz)+'.csv', index=False)


