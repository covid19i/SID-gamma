from __future__ import division, print_function
import os, time
import numpy as np
import rawpy
import glob
from pip._vendor.pkg_resources import null_ns_handler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt



dic_y = {(0.01,0.01):0,(0.01,0.03):1, (0.01,0.06):2,
         (0.08,0.01):3, (0.08,0.03):4,(0.08,0.06):5,
         (0.16,0.01):6,(0.16,0.03):7,(0.16,0.06):8 }

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)

    return out

def add_noise(img):
    #sigma_s = [0.0, 0.0, 0.0]  #0-0.16
    #sigma_c = [0.005, 0.005, 0.005]  #0-0.06

    img = img/100
    w, h, c = img.shape


    #sigma_s = np.random.uniform(0.0, 0.16, (4,))
    #sigma_c = np.random.uniform(0.0, 0.06, (4,))
    #changing to classification
    sig_s = np.random.choice([0.01,0.08,0.16])
    sig_c = np.random.choice([0.01,0.03,0.06])

    sigma_s = [sig_s]*4
    sigma_c = [sig_c]*4

    #add ns
    #sigma_s: 0-0.16
    sigma_s = np.reshape(sigma_s, (1, 1, c))
    noise_s_map = np.multiply(sigma_s, img)
    noise_s = np.random.randn(w, h, c) * noise_s_map
    img = img + noise_s

    #add nc
    #sigma_c:0.01-0.06
    noise_c = np.zeros((w, h, c))
    for chn in range(4):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h))
    img = img + noise_c


    return img, sig_s, sig_c


path_sony = "./Sony/long"
#path_fuji = "./Fuji/long"

# "0" for training set and "2" for validation set, 1- test

## Read train data
#shape -> 4240*2832 , 6000*4000

train_x = list()
train_y_s = list()
train_y_c = list()
train_y = list()

for i in glob.glob(path_sony + '/0*.ARW'):
    print("reading: "+i)

    raw = rawpy.imread(i)
    raw = pack_raw(raw)

    raw, sigma_s, sigma_c = add_noise(raw)

    train_x.append(raw)
    train_y_s.append(sigma_s)
    train_y_c.append(sigma_c)
    train_y.append(dic_y[(sigma_s, sigma_c)])

## Read val data

val_x = list()
val_y_c = list()
val_y_s = list()
val_y = list()
for i in glob.glob(path_sony + '/2*.ARW'):
    print("reading: "+i)

    raw = rawpy.imread(i)
    raw = pack_raw(raw)

    raw, sigma_s, sigma_c = add_noise(raw)
    val_x.append(raw)
    val_y_s.append(sigma_s)
    val_y_c.append(sigma_c)
    val_y.append(dic_y[(sigma_s, sigma_c)])

## Read test data

#j = 0
test_x = list()
test_y_s = list()
test_y_c = list()
test_y = list()
for i in glob.glob(path_sony + '/1*.ARW'):
    #j+=1
    #if(j>2):
    #break
    print("reading: "+i)

    raw = rawpy.imread(i)
    raw = pack_raw(raw)

    raw, sigma_s, sigma_c = add_noise(raw)
    test_x.append(raw)
    test_y_s.append(sigma_s)
    test_y_c.append(sigma_c)
    test_y.append(dic_y[(sigma_s, sigma_c)])

#train
train_x = torch.from_numpy(np.array(train_x, dtype=np.float32))#.float()
#train_x = torch.from_numpy(train_x)
n, x, y, c = train_x.shape
train_x = torch.reshape(train_x, (n,c,x,y))
train_y = torch.from_numpy(np.array(train_y, dtype=np.int64))#.float()

#val
val_x = torch.from_numpy(np.array(val_x, dtype=np.float32))#.float()
#val_x = torch.from_numpy(val_x)
print(val_x.shape)
n, x, y, c = val_x.shape
val_x = torch.reshape(val_x, (n,c,x,y))
val_y = torch.from_numpy(np.array(val_y, dtype=np.int64))#.float()

#test
test_x = torch.from_numpy(np.array(test_x, dtype=np.float32))#.float()
#test_x = torch.from_numpy(test_x)
n, x, y, c = test_x.shape
test_x = torch.reshape(test_x, (n,c,x,y))
test_y = torch.from_numpy(np.array(test_y, dtype=np.int64))#.float()


class NoiseDataset(Dataset):

    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.target[idx]

        if self.transform:
            data = self.transform(data)

        res = (data,label)
        return res


batch_size = 64

data_transforms = transforms.RandomChoice([
    #transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
    #transforms.RandomRotation(180)
])

train_loader = torch.utils.data.DataLoader(
    NoiseDataset(train_x, train_y, transform=data_transforms),
    batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    NoiseDataset(val_x, val_y, transform=data_transforms),
    batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(
    NoiseDataset(test_x, test_y, transform=data_transforms),
    batch_size=batch_size, shuffle=True, num_workers=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
nclasses = 9


class NetL(nn.Module):

    def __init__(self):
        super(NetL, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 353 * 529, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nclasses)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = NetL()
lr = 0.01
epochs = 50
optimizer = optim.Adam(model.parameters(), lr=lr)
x1 = list()
y1 = list()

def trainL(epoch):
    model.train()
    #print("here1")
    for batch_idx, (data, target) in enumerate(train_loader):
        #print("here2")
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        #model = model.double()
        output = model(data)
        #print("here3")
        #print(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def validationL():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        output = model(data)
        validation_loss += F.cross_entropy(output, target, reduction="sum").item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    y1.append(validation_loss)

for epoch in range(1, epochs + 1):
    trainL(epoch)
    x1.append(epoch)
    validationL()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '.')

plt.xlabel("epoch")

plt.ylabel("Validation Loss")

plt.plot(x1,y1)
plt.show()

print("test_data:")
def testLL():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        validation_loss += F.cross_entropy(output, target, reduction="sum").item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

testLL();





