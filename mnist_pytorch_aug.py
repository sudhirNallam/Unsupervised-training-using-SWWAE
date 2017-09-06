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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)



print('loading data!')
trainset_labeled = pickle.load(open("train_labeled_aug.p", "rb")) ## changed to augmented data
validset = pickle.load(open("validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=32, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
unlabel_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=32, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


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
        x = F.dropout(x, training=self.training)
        x_NLL = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x_NLL))
        x_DECONVII = F.relu(self.fc4(x))
        x = x_DECONVII.view(self.get_size(x_DECONVII),20,4,4)
        x_DECONVI = F.relu(self.deconv1(F.max_unpool2d(x, indices2 , 2, 2)))
        output = F.relu(self.deconv2(F.max_unpool2d(x_DECONVI, indices1 , 2, 2)))
        return output, F.log_softmax(x_NLL)
    
    def get_size(self, x):
        return x.size()[0]

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    crit = nn.MSELoss()
    for label, unlabel in zip(enumerate(train_loader), enumerate(unlabel_loader)):
        batch_idx, (data, target) = label
        n = data.size()[0]
        batch_idx, (data_un, target_un) = unlabel
        data = torch.cat((data, data_un), 0)
        target = torch.cat((target, torch.LongTensor(32).fill_(-1)), 0)
        #Convert to variables
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        out_decode, output = model(data)
        loss_delta = crit(out_decode, data)
        loss = F.nll_loss(output[0:n], target[0:n]) + loss_delta
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 2*len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch, valid_loader, dataset_label = 'Test'):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        _, output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\n' + dataset_label + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return [test_loss, correct]

loss_vec = []
correct_vec = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    temp = test(epoch, valid_loader)
    loss_vec.append(temp[0])
    correct_vec.append(temp[1])

import datetime
timestr = datetime.datetime.now().strftime("%b%d").lower()
loss_filename = "results/" + timestr + "_" + str(args.epochs) + "epochs_val_loss.txt"
correct_filename = "results/" + timestr + "_" + str(args.epochs) + "epochs_val_correct.txt"
with open(loss_filename, "w") as thefile:
    thefile.write(",".join(map(str, loss_vec)))
with open(correct_filename, "w") as thefile:
    thefile.write(",".join(map(str, correct_vec)))

#outfile.write("\n".join(itemlist))