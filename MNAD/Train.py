import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random
import argparse
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from glob import glob

#change dir to MNAD
os.chdir('/home/smoothjazzuser/videogame-anomoly/MNAD')

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=30, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=12, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.15, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.15, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=84, help='height of input images')#256
parser.add_argument('--w', type=int, default=84, help='width of input images')#256
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=15, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='bugs', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

downscale = True
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=False, num_workers=args.num_workers_test, drop_last=False)
 
from model.Reconstruction import *
model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)
params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()


# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

if downscale:
    def loss_func_mse(x, y):
        x, y = F.interpolate(x, size=(args.h, args.w)), F.interpolate(y, size=(args.h, args.w)) #mode = nearest-exact
        #fourier transform
        x = x.squeeze()
        y = y.squeeze()

        #to numpy
        if False:
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            xx = x.copy()
            yy = y.copy()
            #save original dtype
            orig_dtype = x.dtype
            #convert to complex dtype
            xx = xx.astype(np.complex64)
            yy = yy.astype(np.complex64)
            for dim in range(args.c):
                xx[dim] = np.fft.fft2(x[dim], axes=(0,1))
                yy[dim] = np.fft.fft2(y[dim], axes=(0,1))
                xx[dim] = np.fft.fftshift(x[dim])
                yy[dim] = np.fft.fftshift(y[dim])

                #low freqs are less than .1
                xx[dim][xx[dim] > .1] = 0
                yy[dim][yy[dim] > .1] = 0
                xx[dim] = np.fft.ifftshift(xx[dim])
                yy[dim] = np.fft.ifftshift(yy[dim])
                xx[dim] = np.fft.ifft2(xx[dim], axes=(0,1))
                yy[dim] = np.fft.ifft2(yy[dim], axes=(0,1))
            
            #to tensor
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()
            #cast back to real dtype
            xx = xx.astype(orig_dtype)
            yy = yy.astype(orig_dtype)
            xx = torch.from_numpy(xx).cuda()
            yy = torch.from_numpy(yy).cuda()

            loss = F.mse_loss(x,y) + F.mse_loss(yy,xx)
        # loss = F.mse_loss(x,y) * F.mse_loss(xx,yy) * F.mse_loss(ax,ay)
        # loss = F.mse_loss(x,y) * F.mse_loss(xx,yy)
        else:
            loss = F.mse_loss(x,y)
        return loss
else:
    loss_func_mse = nn.MSELoss(reduction='none')

# Training

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

for epoch in tqdm(range(args.epochs)):
    labels_list = []
    model.train()
    
    start = time.time()
    kkk = 0
    #tqdm enumerate
    for j,(imgs) in tqdm(enumerate(train_batch), total=len(train_batch)):
        imgs = Variable(imgs).cuda()
        outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    if args.method == 'pred':
        print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    else:
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    
print('Training is finished')
torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
    
sys.stdout = orig_stdout
f.close()



