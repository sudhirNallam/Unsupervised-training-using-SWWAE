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
import os
from sub import subMNIST
from scipy import ndimage
from scipy import stats
import random
import datetime


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
parser.add_argument('--saveLocation', type=str, metavar='S', required=True,
                    help='Location for saving trained models')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
## define a function that takes a larger than dim x dim image 
## and trims (~)symmetrically to dim x dim.
def trim_image(image, dim):
    ## calculate how many pixels need trimming off.
    temp = [(x - 28)/2.0 for x in image.shape]

    ## just trim the array
    ## can't use negative indices for the case of 0 x[-0] === x[0]
    image_out = image[round(temp[0]):(round(temp[0])+28), round(temp[1]):(round(temp[1])+28)]

    return image_out

## define some functions to zoom, translate and rotate
def zoom_image(image_in):
    ## zooming first
    ## don't want to remove more than 4 from each size, (28+8)/28, 
    ## 1.3 gets rounded down to zoomed size 36
    ## half normal dist center at 1, std 0.15, capped at 1.3
    scale = stats.halfnorm.rvs(loc=1, scale=0.15, size = 2)
    scale = np.clip(scale, 1.0, 1.3)

    ## zoom using scipy.ndimage.zoom
    ## then use custom trim_image to 28 x 28
    image_out = torch.ByteTensor(trim_image(ndimage.zoom(image_in.numpy(), scale), 28))
    return image_out

def translate_image(image_in):
    ## Shifting / translation
    ## no more than 4, 
    ## normal dist with mean 0, sd 2, clipped at +/-4?
    scale = np.random.normal(loc=0, scale=2, size = 2)
    scale = np.clip(scale, -4.0, 4.0)

    ## zoom using scipy.ndimage.shift
    ## don't need to trim image
    image_out = torch.ByteTensor(ndimage.shift(image_in.numpy(), scale))
    return image_out

def rotate_image(image_in):
    ## Rotating
    ## no more than +/- 30
    ## normal dist with mean 0, sd 15, ceiling at 30?
    ## returns larger than 28 x 28 images, need to trim down
    scale = np.random.normal(0, 15, size=1)
    scale = np.clip(scale, -30.0, 30.0)

    ## zoom using scipy.ndimage.rotate
    ## then use custom trim_image to 28 x 28
    image_out = torch.ByteTensor(trim_image(ndimage.rotate(image_in.numpy(), scale), 28))
    return image_out

def dataLoader():
    transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])
    trainset_original = datasets.MNIST('../data', train=True, download=True,
                                      transform=transform)
    train_label_index = []
    valid_label_index = []
    for i in range(10):
        train_label_list = trainset_original.train_labels.numpy()
        label_index = np.where(train_label_list == i)[0]
        label_subindex = list(label_index[:300])
        valid_subindex = list(label_index[300: 1000 + 300])
        train_label_index += label_subindex
        valid_label_index += valid_subindex

    #Train Set
    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub = torch.from_numpy(trainset_np[train_label_index])
    train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

    trainset_new = subMNIST(root='./data', train=True, download=True, transform=transform, k=3000)
    trainset_new.train_data = train_data_sub.clone()
    trainset_new.train_labels = train_labels_sub.clone()

    pickle.dump(trainset_new, open("train_labeled.p", "wb" ))


    #### Augmenting training set
    ## initialize trainset as usual
    trainset_aug = subMNIST(root='./data', train=True, download=True, transform=transform, k=30000)
    ## turns out you can just repeat a tensor, cool
    ## http://pytorch.org/docs/tensors.html#torch.Tensor.repeat
    trainset_aug.train_data = train_data_sub.clone().repeat(10,1,1) ## 4 in the first axis
    print(train_data_sub.size())
    print(trainset_aug.train_data.size())
    trainset_aug.train_labels = train_labels_sub.clone().repeat(10) ## only one axi
    print(train_labels_sub.size())
    print(trainset_aug.train_labels.size())
    ## dims look correct!

    ## load scipy image tools and distributions for sampling
    from scipy import ndimage
    from scipy import stats
    import random
    random.seed(1337)
    #from math import ceil, floor

    ## iterate through and augment
    n = trainset_aug.train_data.size()[0]/10
    print(n)
    iter_vals = range(0, n)
    #iter_vals = range(0, 5)
    ''' 
    for i in iter_vals:
        image_in = trainset_aug.train_data[i]
        trainset_aug.train_data[i+n] = zoom_image(image_in)
        trainset_aug.train_data[i+(n*2)] = translate_image(image_in)
        trainset_aug.train_data[i+(n*3)] = rotate_image(image_in)           
    '''
    for i in iter_vals:
        image_in = trainset_aug.train_data[i]
        trainset_aug.train_data[i+n] = zoom_image(image_in)
        trainset_aug.train_data[i+(n*2)] = zoom_image(image_in)
        trainset_aug.train_data[i+(n*3)] = zoom_image(image_in)
        trainset_aug.train_data[i+(n*4)] = translate_image(image_in)
        trainset_aug.train_data[i+(n*5)] = translate_image(image_in)
        trainset_aug.train_data[i+(n*6)] = translate_image(image_in)
        trainset_aug.train_data[i+(n*7)] = rotate_image(image_in)
        trainset_aug.train_data[i+(n*8)] = rotate_image(image_in)
        trainset_aug.train_data[i+(n*9)] = rotate_image(image_in)


    ## dump to pickle
    pickle.dump(trainset_aug, open("train_labeled_aug.p", "wb" ))

    #Validation Set
    validset_np = trainset_original.train_data.numpy()
    validset_label_np = trainset_original.train_labels.numpy()
    valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
    valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])

    validset = subMNIST(root='./data', train=False, download=True, transform=transform, k=10000)
    validset.test_data = valid_data_sub.clone()
    validset.test_labels = valid_labels_sub.clone()

    pickle.dump(validset, open("validation.p", "wb" ))

    #Unlabeled Data
    train_unlabel_index = []
    for i in range(60000):
        if i in train_label_index or i in valid_label_index:
            pass
        else:
            train_unlabel_index.append(i)

    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
    #train_labels_sub_unl = torch.from_numpy(trainset_label_np[train_unlabel_index])
    temp = np.empty(47000)
    temp.fill(-1)
    train_labels_sub_unl = torch.from_numpy(temp) 

    trainset_new_unl = subMNIST(root='./data', train=True, download=True, transform=transform, k=47000)
    trainset_new_unl.train_data = train_data_sub_unl.clone()
    trainset_new_unl.train_labels = train_labels_sub_unl.clone()

    pickle.dump(trainset_new_unl, open("train_unlabeled.p", "wb" ))

dataLoader()    

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('loading data!')
trainset_labeled = pickle.load(open("train_labeled_aug.p", "rb")) ## changed to augmented data
validset = pickle.load(open("validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
#testset = pickle.load(open("test.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=32, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
unlabel_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=32, shuffle=True)
#test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

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


if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    crit = nn.MSELoss()
    batch_loss = []
    for label, unlabel in zip(enumerate(train_loader), enumerate(unlabel_loader)):
        batch_idx, (data, target) = label
        n = data.size()[0]
        batch_idx, (data_un, target_un) = unlabel
        data = torch.cat((data, data_un), 0)
        target = torch.cat((target, torch.LongTensor(32).fill_(-1)), 0)
        #Convert to variables
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        x_CONVI, x_DECONVI, x_CONVII, x_DECONVII,out_decode, output = model(data)
        #x_CONVII = x_CONVII.detach()
        #x_DECONVII = x_DECONVII.detach()
        #x_CONVI = x_CONVI.detach()
        #x_DECONVI = x_DECONVI.detach()
        #loss_delta = crit(out_decode, data) + crit(x_DECONVII, x_CONVII) + crit(x_DECONVI, x_CONVI)
        loss_delta = crit(out_decode, data)
        loss = F.nll_loss(output[0:n], target[0:n]) + loss_delta
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data[0])
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 2*len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return np.mean(batch_loss)

def test(epoch, valid_loader, dataset_label = 'Test'):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        _,_,_,_,_,output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return [test_loss, correct]		

def test_file(test_loader, timestr, epochs):
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
    predict_label.to_csv('sample_submission'+timestr+'_'+str(epochs)+'.csv', index=False)

train_loss_vec = []
loss_vec = []
correct_vec = []
for epoch in range(1, args.epochs + 1):
    temp_train = train(epoch)
    temp = test(epoch, valid_loader)
    loss_vec.append(temp[0])
    correct_vec.append(temp[1])
    train_loss_vec.append(temp_train)
    if epoch >= 30 and epoch % 10 ==0:
        torch.save(model.state_dict(), os.getcwd() + args.saveLocation+'_'+str(epoch)+'.t7')



filename = "results/"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
timestr = datetime.datetime.now().strftime("%b%d").lower()
loss_filename = "results/" + timestr + "_" + str(args.epochs) + "epochs_val_loss.txt"
correct_filename = "results/" + timestr + "_" + str(args.epochs) + "epochs_val_correct.txt"
train_filename = "results/" + timestr + "_" + str(args.epochs) + "epochs_train_loss.txt"
with open(loss_filename, "w") as thefile:
    thefile.write(",".join(map(str, loss_vec)))
with open(correct_filename, "w") as thefile:
    thefile.write(",".join(map(str, correct_vec)))
with open(train_filename, "w") as thefile:
    thefile.write(",".join(map(str, train_loss_vec)))

#test_file(test_loader, timestr, args.epochs)
