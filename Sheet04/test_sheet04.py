import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import random
import numpy as np
from collections import defaultdict
import json
import _pickle as pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = '  '
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ShallowModel(nn.Module):
    ### TODO ####
    def __init__(self):
        super(ShallowModel,self).__init__()
        self.numClasses = 10
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),stride=(1,1))
        self.mpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features= 2880, out_features=self.numClasses)
        self.shallow_conv = nn.Sequential(self.conv1,self.relu,self.conv2,self.relu,self.mpool)
    def forward(self,x):
        x_1 = self.shallow_conv(x)
        x_2 = torch.flatten(x_1,1)
        output = self.lin1(x_2)

        return output



class WiderModel(nn.Module):
    ### TODO ####
    def __init__(self):
        super(WiderModel, self).__init__()
        self.numClasses =10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1))  ## in_channels=10
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1))  ## in_channels=10
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=(1,1)) ## in_channels=10
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6760,self.numClasses)
    def forward(self,x):
        x_1 = self.relu(self.conv1(x))
        x_2 = self.relu(self.conv2(x))
        x_3 = self.relu(self.conv3(x))
        x_4 = self.relu(self.conv4(x))

        x_cat = torch.cat((x_1,x_2,x_3,x_4),dim=1)
        out = self.mpool(x_cat)
        # print("Output of Wider model: ",out.shape) ## [64, 40, 13, 13]
        out = torch.flatten(out,1)
        # print("Output of Wider model: ",out.shape) ## [64, 6760]
        output = self.fc1(out)

        return output


class DeeperModel(nn.Module):
    ### TODO ####
    def __init__(self,batchNorm=False):
        super(DeeperModel,self).__init__()
        self.batch_norm = batchNorm
        self.numClasses = 10
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=10, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), stride=(1,1))
        self.mpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(3,3), stride=(1,1))
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(40)
        self.bn4 = nn.BatchNorm2d(80)
        if self.batch_norm:
            self.convLayer = nn.Sequential(self.conv1,self.bn1,self.conv2,self.bn2, self.mpool1,
                                           self.conv3,self.bn3,self.conv4,self.bn4,self.mpool2)
        else:
            self.convLayer = nn.Sequential(self.conv1,self.conv2,self.mpool1, self.conv3,self.conv4,self.mpool2)

        self.fc1 = nn.Linear(1280,200)
        self.fc2 = nn.Linear(200,self.numClasses)

    def forward(self,x):
        x1 = self.convLayer(x)
        x2 = torch.flatten(x1,1)
        x3 = self.fc1(x2)
        out = self.fc2(x3)

        return out



def main():
    ## Get data
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    ## Training Parameters
    nepochs = 10
    batch_size = 64

    ## Create Dataloaders
    ### TODO ####
    train_dataloader = DataLoader(mnist_trainset,batch_size,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(mnist_testset,batch_size,num_workers=4)


    ### Create Model
    # model = ShallowModel().to(device) ## Test Accuracy : 98.2300%
    # model = WiderModel().to(device) ## with one fc layer:  Test Accuracy : 98.18
    # model = DeeperModel(batchNorm=False).to(device) ## Test Accuracy : 98.4900%
    model = DeeperModel(batchNorm=True).to(device) ## Test Accuracy : 98.5900%

    ### Define Opitmizer
    ### TODO ####
    opti = torch.optim.Adam(model.parameters(),lr = 0.0001)
    criterion = nn.CrossEntropyLoss()
    # ### Train
    ### TODO ####


    train_samples = len(train_dataloader.dataset)
    for i in range(nepochs):
        model.train()
        running_loss=0
        for batch_idx, data in enumerate(train_dataloader):
            img, lbl = data
            img,lbl = img.to(device), lbl.to(device)
            opti.zero_grad()
            out = model(img)
            loss = criterion(out,lbl)
            running_loss += loss.item() * img.shape[0]
            loss.backward()
            opti.step()
        epoch_loss = running_loss/ train_samples
        print("****************************************************")
        print("Epoch:{:d}, Loss : {:.4f}".format(i,epoch_loss))
        print("****************************************************")

    ## Save Model for sharing
    # torch.save(model.state_dict(), './model_shallow')
    # torch.save(model.state_dict(), './model_wider')
    # torch.save(model.state_dict(), './model_deeper')
    # torch.save(model.state_dict(), './model_deeper_bn')


    ### Load Saved model
    # model.load_state_dict(torch.load("./model_deeper_bn")) ##  model_shallow  model_wider   model_deeper  model_deeper_bn
    ## Access and save weights of the model
    weight_dict = defaultdict(list)
    for name,param in model.named_parameters():
        weight_dict['param_name'].append(name)
        weight_dict['param_shape'].append(param.shape)
        weight_dict['param_value'].append(param.data)

    ## shallow_model_weights.txt   wider_model_weights.txt   deeper_model_weights.txt   deeperBn_model_weights.txt
    with open("deeperBn_model_weights.txt",'wb')as out_file:
        pickle.dump(weight_dict,out_file)

    ### Load weight dictionary
    # with open("deeperBn_model_weights.txt","rb") as in_file:
    #     wt_dict = pickle.load(in_file)
    # print(wt_dict)
    ## Test
    ### TODO ####
    model.eval()
    correct = 0
    total_samples = len(test_dataloader.dataset)
    with torch.no_grad():
        for batch_idx,data in enumerate(test_dataloader):
            img, lbl = data
            img, lbl = img.to(device),lbl.to(device)
            out = model(img)
            _,pred = torch.max(out,dim=1)
            correct += torch.sum(pred==lbl)
        accuracy = correct / total_samples * 100
        print("****************************************************")
        print("Test Accuracy : {:.4f}%".format(accuracy))
        print("****************************************************")




if __name__ == '__main__':
    main()
