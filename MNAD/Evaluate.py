if True:
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
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    from model.Reconstruction import *
    from sklearn.metrics import roc_auc_score
    from utils import *
    import random
    import glob

    import argparse
############################################
    
    loss_sections = 4
    std_loss_correction = {i:[] for i in range(loss_sections)} #False
    method = 'rolling_std'
    force_keys = False
    prev = 0.0000001
    import matplotlib.pyplot as plt
    from time import sleep
    def normalize_array(array:np.ndarray):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array-min_val)/(max_val-min_val)#*255
    import matplotlib.image as mpimg
    import shutil
    import cv2
    from scipy.stats import percentileofscore
    import PIL
############################################
    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=1, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--th', type=float, default=0.0001, help='threshold for test updating')#1.5e-09
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='bugs', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    args = parser.parse_args()
    parser.add_argument('--model_dir', type=str, default=f'./exp/{args.dataset_type}/{args.method}/log/model.pth', help='directory of model')
    parser.add_argument('--m_items_dir', type=str, default=f'./exp/{args.dataset_type}/{args.method}/log/keys.pt',help='directory of model')

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

    test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)
print("The number of test data is %d" % test_size)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

if not std_loss_correction:
    loss_func_mse = nn.MSELoss(reduction='none')
else:
    mse = nn.MSELoss(reduction='none')
    def loss_func_mse(i, t):
        global loss_sections, std_loss_correction
        #split l into len(loss_sections) sections
        losses = {'i':[], 't':[]}

        #split tensor into sections
        for section in range(loss_sections):
            losses['i'].append(i[:,section*(i.shape[1]//loss_sections):(section+1)*(i.shape[1]//loss_sections)])
            losses['t'].append(t[:,section*(t.shape[1]//loss_sections):(section+1)*(t.shape[1]//loss_sections)])

        loss = 0
        for section in range(loss_sections): 
            l = mse(losses['i'][section], losses['t'][section])
            l = l.mean()
            std_loss_correction[section].append(l.cpu().detach().numpy())
            if method =='rolling_std':
                window = len(std_loss_correction[section])
                if window > 100:
                    window = 100
                p = np.std(std_loss_correction[section][-window:]) + 0.0000001
                loss = l * abs(.5 - p)
            elif method == 'percentile':
                p = percentileofscore(std_loss_correction[section], l.cpu().detach().numpy())/100  + 0.000000001
                loss += l*abs((.5 - p)**2)
            elif method == 'mean':
                #l = l / np.mean(std_loss_correction[section])
                p = abs(np.mean(std_loss_correction[section]) - l.cpu().detach().numpy()) + 0.000000001
                loss += l*p


        return loss

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

videos = OrderedDict()

videos_list = sorted(glob.glob(os.path.join(test_folder, '*')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = sorted(glob.glob(os.path.join(video, '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    #videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list, key=lambda x: int(x.split('/')[-1].split('.')[0])):
#for video in videos_list:
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    else:
        labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()
kkk=0
ccc = int(test_size/args.test_batch_size)
diffs = []
preds = []
ground_truths = []
label_list = np.load(f"/home/smoothjazzuser/videogame-anomoly/MNAD/dataset/frame_labels_{args.dataset_type}.npy", allow_pickle=True).tolist()[0]
loss_hist = {i:[] for i in range(loss_sections)}
for k,(imgs) in enumerate(test_batch):
    
    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':

        if force_keys:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items.clone(), False)#m_items_test
        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)#
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])

        # visualize the difference between reconstructed and predicted frames in image format
        #diff = np.abs((outputs[0].detach().cpu().numpy()+1)/2 - (imgs[0].detach().cpu().numpy()+1)/2).transpose(1,2,0)
        diff = np.abs((outputs[0].detach().cpu().numpy()+1)/2 - (imgs[0,3*4:].detach().cpu().numpy()+1)/2).transpose(1,2,0)
        #diff = normalize_array(diff)
        diffs.append(diff)
        pred = ((outputs[0].detach().cpu().numpy()+1)/2).transpose(1,2,0)
        #pred = normalize_array(pred)
        preds.append(pred)

    
    else:
        if force_keys:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items.clone(), False)#

        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

         # visualize the difference between reconstructed and predicted frames in image format
        diff = np.abs((outputs[0].detach().cpu().numpy()+1)/2 - (imgs[0].detach().cpu().numpy()+1)/2).transpose(1,2,0)
        #diff = normalize_array(diff)
        diffs.append(diff)

        pred = (outputs[0].detach().cpu().numpy()+1)/2
        pred = pred.transpose(1,2,0)
        preds.append(pred)


    if  point_sc < args.th: # or k < 5: 
        print(f"updating memory -GT {label_list[kkk]} -th {args.th} -value {point_sc}" )
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)
    prev = point_sc

    kkk+=1
   

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
    print(f"pairs pred: {round(100*kkk/ccc,3)}%, mse_feas:{mse_feas}", end = "\r")


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

np.save(f"./exp/{args.dataset_type}/{args.method}/log/anomaly_score_total_list.npy", anomaly_score_total_list)
np.save(f"./exp/{args.dataset_type}/{args.method}/log/anomaly_score_total_list.npy", anomaly_score_total_list)
np.save(f"./exp/{args.dataset_type}/{args.method}/log/psnr_list.npy", psnr_list)
np.save(f"./exp/{args.dataset_type}/{args.method}/log/feature_distance_list.npy", feature_distance_list)
np.save(f"./exp/{args.dataset_type}/{args.method}/log/auc.npy", accuracy)
np.save(f"./exp/{args.dataset_type}/{args.method}/log/labels_list.npy", labels_list)

if os.path.exists(f"./exp/{args.dataset_type}/{args.method}/log/diffs/"):
    l = f"./exp/{args.dataset_type}/{args.method}/log/diffs/"
    shutil.rmtree(l)
if os.path.exists(f"./exp/{args.dataset_type}/{args.method}/log/preds/"):
    l = f"./exp/{args.dataset_type}/{args.method}/log/preds/"
    shutil.rmtree(l)
if os.path.exists(f"./exp/{args.dataset_type}/{args.method}/log/ground_truths/"):
    l = f"./exp/{args.dataset_type}/{args.method}/log/ground_truths/"
    shutil.rmtree(l)

if not os.path.exists(f"./exp/{args.dataset_type}/{args.method}/log/diffs/"):
    os.makedirs(f"./exp/{args.dataset_type}/{args.method}/log/diffs/")
if not os.path.exists(f"./exp/{args.dataset_type}/{args.method}/log/preds/"):
    os.makedirs(f"./exp/{args.dataset_type}/{args.method}/log/preds/")

for (i,img) in enumerate(preds): 
    #use pillow to save the image in grayscale
    if args.c == 1:
        img = img*255
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = PIL.Image.fromarray(img, 'L')
        img.save(f"./exp/{args.dataset_type}/{args.method}/log/preds/{i}.jpg")
    else:
        img = img*255
        img = img.astype(np.uint8)
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(f"./exp/{args.dataset_type}/{args.method}/log/preds/{i}.jpg")

for (i,img) in enumerate(diffs): 
    if args.c == 1:
        img = img*255
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        img = PIL.Image.fromarray(img, 'L')
        img.save(f"./exp/{args.dataset_type}/{args.method}/log/diffs/{i}.jpg")
    else:
        img = img*255
        img = img.astype(np.uint8)
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(f"./exp/{args.dataset_type}/{args.method}/log/diffs/{i}.jpg")


print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')
