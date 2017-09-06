from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
#from mnist_pytorch import Net


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--savedLocation', type=str, required = True, metavar='N',
                    help='Location of the model')
parser.add_argument('--resultsLocation', type=str, required = True, metavar='N',
                    help='Location of the results')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



print('loading data!')
testset = pickle.load(open("test.p", "rb"))

test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # Deconvolution
        self.fc3 = nn.Linear(10, 50)
        self.fc4 = nn.Linear(50, 320)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        #Loss
        self.loss_fn = nn.MSELoss()
        

    def forward(self, x):
        x, indices1 = F.max_pool2d(self.conv1(x), 2, return_indices=True) 
        x_CONVI = F.relu(x)
        x, indices2 = F.max_pool2d(self.conv2_drop(self.conv2(x_CONVI)), 2, return_indices=True)
        x_CONVII = F.relu(x)
        x = x_CONVII.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x_NLL = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x_NLL))
        x_DECONVII = F.relu(self.fc4(x))
        x = x_DECONVII.view(self.get_size(x_DECONVII),20,4,4)
        x_DECONVI = F.relu(self.deconv1(F.max_unpool2d(x, indices2 , 2, 2)))
        output = F.relu(self.deconv2(F.max_unpool2d(x_DECONVI, indices1 , 2, 2)))
        return x_CONVI, x_CONVII, x_DECONVII, x_DECONVI, output, F.log_softmax(x_NLL)
    
    def get_size(self, x):
        return x.size()[0]


model = Net()
model.load_state_dict(torch.load(os.getcwd() + args.savedLocation))


def test_file(test_loader):
    label_predict = np.array([])
    model.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        _,_,_,_,_,output = model(data)
        temp = output.data.max(1)[1].numpy().reshape(-1)
        label_predict = np.concatenate((label_predict, temp))
    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    predict_label.to_csv(os.getcwd() + args.resultsLocation, index=False)

test_file(test_loader)
