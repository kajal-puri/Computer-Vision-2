import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '  '

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ShallowModel(nn.Module):
    ### TODO ####

class WiderModel(nn.Module):
    ### TODO ####


class DeeperModel(nn.Module):
    ### TODO ####

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


    ### Create Model
    # model = ShallowModel()
    # model = WiderModel()
    # model = DeeperModel(batchNorm=False)
    # model = DeeperModel(batchNorm=True)

    ### Define Opitmizer
    ### TODO ####

    # ### Train
    ### TODO ####

    ### Save Model for sharing
    # torch.save(model.state_dict(), './model')

    ## Test
    ### TODO ####



if __name__ == '__main__':
    main()
