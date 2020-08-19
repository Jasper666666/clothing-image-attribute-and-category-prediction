import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
import time
import argparse
import ast
from dataset import DeepFashion,DeepFashion_landmark
from torch.nn import DataParallel
#from sync_batchnorm import DataParallelWithCallback as DataParallel
import numpy as np
#from resnet import *
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from model import model_alexnet_v2,model_vgg16_v2,model_vgg16_all_v1,model_vgg16_all_v2
from evaluator import evaluator,evaluator_v2
from another_eval import another_Evaluator
import random
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train_test', help="train or test or train_test", type=str)
parser.add_argument('--ckpt_path', default="/data/hsy/model_vgg16_all_v1_16.ckpt", help="path of ckpt file", type=str)
parser.add_argument('--start_ckpt', default="/data/hsy/model_vgg16_all_v1.ckpt", help="the start of ckpt", type=str)
parser.add_argument('--epochs', default=3, help="epochs", type=int)
parser.add_argument('--batch_size', default=16, help="batch_size", type=int)




def train(net, trainloader, epochs,start_ckpt,save_path,valloader,adj,lambda_lm=1,lambda_c=1,lambda_a=1):
    if start_ckpt!="":
        net.load_state_dict(torch.load(start_ckpt))

    criterion_landmark=nn.MSELoss().cuda()
    criterion_attribute = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.])).cuda()
    criterion_category = nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = torch.optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,verbose=True)


    for epoch in range(epochs):
        net.train()
        t = time.time()
        tq = tqdm(trainloader, ncols=80, ascii=True)
        correct_category = 0
        correct_attribute=0
        total = 0
        total_a_loss=0.
        total_c_loss=0.
        total_lm_loss=0.
        for i, batch in enumerate(tq):
            image, label ,landmarks,category,landmark_map= batch['image'], batch['label'],batch['landmarks'],batch['category'],batch['landmark_map']
            image = image.cuda()
            label=label.long()
            label = label.cuda()
            landmarks=landmarks.cuda()
            category=category.cuda()
            landmark_map=landmark_map.cuda()

            ret=net(image,landmarks,adj)
            pre_category=ret['category_output']
            pre_attribute =ret['attr_output']
            pre_landmark_map=ret['lm_pos_map']
            #pre_category,pre_attribute = net(image,landmarks,adj)


            lm_loss=lambda_lm*criterion_landmark(pre_landmark_map,landmark_map)
            a_loss=lambda_a*criterion_attribute(pre_attribute,label)
            c_loss=lambda_c*criterion_category(pre_category, category)
            loss = lm_loss+a_loss+c_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += image.size(0)

            _, pre_category_re = torch.max(pre_category, 1)
            correct_category += (pre_category_re == category).sum().item()


            total_a_loss+=a_loss.item()
            total_c_loss += c_loss.item()
            total_lm_loss+=lm_loss.item()


        print(epoch+1,'/',epochs,'train_c:',float(correct_category)/total,'total_lm_loss',total_lm_loss,'loss_c:',total_c_loss,'loss_a:',total_a_loss)
        torch.save(net.state_dict(), save_path)
        #val-----------------------------------------------------------
'''
        net.eval()
        tq = tqdm(valloader, ncols=80, ascii=True)
        correct_category = 0
        correct_attribute = 0
        total = 0
        for i, batch in enumerate(tq):
            image, label, landmarks, category = batch['image'], batch['label'], batch['landmarks'], batch['category']
            image = image.cuda()
            label = label.cuda()
            landmarks = landmarks.cuda()
            category = category.cuda()

            ret = net(image, landmarks, adj)
            pre_category = ret['category_output']
            pre_attribute = ret['attr_output']
            pre_landmark_map = ret['lm_pos_map']

            total += image.size(0)

            _, pre_category_re = torch.max(pre_category, 1)
            correct_category += (pre_category_re == category).sum().item()



        print('val_c:',float(correct_category)/total,time.time()-t)
 '''


def test(net, testloader, load_path,adj):
    net.load_state_dict(torch.load(load_path))
    net.eval()
    evaluator = another_Evaluator(category_topk=(1, 3, 5), attr_topk=(3, 5))
    t=time.time()
    correct_category = 0
    correct_attribute = 0
    total = 0
    c_topk = np.zeros((2))  # number of top-3 and top-5
    with torch.no_grad():
        tq = tqdm(testloader, ncols=80, ascii=True)
        for i, batch in enumerate(tq):
            image, label, landmarks, category = batch['image'], batch['label'], batch['landmarks'], batch['category']
            image = image.cuda()
            label = label.cuda()
            landmarks = landmarks.cuda()
            category = category.cuda()

            output = net(image, landmarks, adj)
            pre_category = output['category_output']
            pre_attribute = output['attr_output']
            pre_landmark_map = output['lm_pos_map']

            total += image.size(0)
            # evaluate category===================================================================
            _, pre_category_re = torch.max(pre_category, 1)
            correct_category += (pre_category_re == category).sum().item()

            _, pre_category_re = pre_category.topk(5, dim=1, largest=True, sorted=True)
            for j in range(image.size(0)):
                if pre_category_re[j][0] == category[j] or pre_category_re[j][1] == category[j] or pre_category_re[j][2] == category[j]:
                    c_topk[0] += 1
                    c_topk[1] += 1
                    continue
                if pre_category_re[j][3] == category[j] or pre_category_re[j][4] == category[j]:
                    c_topk[1] += 1

            sample={}
            sample['category_label']=category
            sample['attr']=label
            evaluator.add(output, sample)

        print('test_c:', float(correct_category) / total)
        print('c_top3:', float(c_topk[0]) / total, 'c_top5:', float(c_topk[1]) / total)
        ret = evaluator.evaluate()
        for topk, accuracy in ret['category_accuracy_topk'].items():
            print('metrics/category_top{}'.format(topk), accuracy)


        for topk, accuracy in ret['attr_group_recall'].items():
            for attr_type in range(1, 6):
                print('metrics/attr_top{}_type_{}_{}_recall'.format(
                    topk, attr_type, attr_type), accuracy[attr_type - 1]
                )

            print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])

def draw_lm(net, testset, load_path,adj):
    net.load_state_dict(torch.load(load_path))
    net.eval()
    t=time.time()
    with torch.no_grad():
        for iii in range(50):
            index=random.randint(0,len(testset)-1)
            img_path=testset[index]['img_path']
            image=testset[index]['image']
            landmarks=testset[index]['landmarks']
            x1=testset[index]['x1']
            y1 = testset[index]['y1']
            x2 = testset[index]['x2']
            y2 = testset[index]['y2']
            xx=x2-x1
            yy=y2-y1

            image=image.cuda()
            image=image.view(1,3,224,224)
            landmarks=landmarks.cuda()
            landmarks=landmarks.view(1,8,4)
            output = net(image, landmarks, adj)
            pre_landmark_map = output['lm_pos_map']
            pre_landmark_pos=output['lm_pos_output']
            #pre_landmark_map, pre_landmark_pos = net(image)
            pre_landmark_map=pre_landmark_map[0]
            pre_landmark_pos=pre_landmark_pos[0]
            landmarks=landmarks[0]
            #print(pre_landmark_map.size())
            #print(pre_landmark_pos.shape)

            pre_img=cv2.imread(img_path)
            real_img=cv2.imread(img_path)
            both_img=cv2.imread(img_path)

            for i in range(landmarks.size(0)):
                if landmarks[i][1]==1:
                    x=landmarks[i][2]
                    y=landmarks[i][3]
                    x=x/224.
                    y=y/224.
                    x=int(x*xx+x1)
                    y=int(y*yy+y1)
                    cv2.circle(real_img, (x, y), 5, (0, 255, 0), -1)
                    cv2.circle(both_img, (x, y), 5, (0, 255, 0), -1)

            for i in range(pre_landmark_pos.shape[0]):
                x = pre_landmark_pos[i][0]
                y = pre_landmark_pos[i][1]
                #---------------------------------------------------------------------------
                #if pre_landmark_map[i][int(x)][int(y)]<0.5:
                 #   continue
                # ---------------------------------------------------------------------------
                #print(i, x, y)
                x = x / 224.
                y = y / 224.
                x = int(x * xx + x1)
                y = int(y * yy + y1)
                #print(i,x,y)
                cv2.circle(pre_img, (x, y), 5, (255, 0, 0), -1)
                cv2.circle(both_img, (x, y), 5, (255, 0, 0), -1)
            cv2.imwrite('./lm_img_all/'+str(iii)+'_pre.jpg', pre_img)
            cv2.imwrite('./lm_img_all/' + str(iii) + '_real.jpg', real_img)
            cv2.imwrite('./lm_img_all/' + str(iii) + '_both.jpg', both_img)

if __name__ == '__main__':
    args = parser.parse_args()
    mode = args.mode
    ckpt_path = args.ckpt_path
    epochs=args.epochs
    start_ckpt=args.start_ckpt
    batch_size=args.batch_size

    trainset = DeepFashion_landmark("test")
    valset=DeepFashion_landmark("val")
    testset = DeepFashion_landmark("test")
    trainloader = DataLoader(trainset, batch_size,shuffle=True)
    valloader=DataLoader(valset, batch_size)
    testloader = DataLoader(testset, batch_size)

    net = model_vgg16_all_v2(num_category=50,num_attribute=1000).cuda()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    adj = torch.zeros((8, 8))
    adj[0][2] = 1.
    adj[2][0] = 1.
    adj[1][2] = 1.
    adj[2][1] = 1.
    adj[3][2] = 1.
    adj[2][3] = 1.
    adj[5][2] = 1.
    adj[2][5] = 1.
    adj[5][4] = 1.
    adj[4][5] = 1.
    adj[5][6] = 1.
    adj[6][5] = 1.
    adj[5][7] = 1.
    adj[7][5] = 1.
    adj = adj.cuda()

    # print(net)
    if mode == "train_test":
        train(net, trainloader, epochs,start_ckpt,ckpt_path,valloader,adj,lambda_lm=100,lambda_c=1,lambda_a=20)
        test(net, testloader, ckpt_path,adj)
        #draw_lm(net, testset, ckpt_path, adj)
    elif mode == "train":
        train(net, trainloader, epochs,start_ckpt,ckpt_path,valloader,adj,lambda_lm=100,lambda_c=1,lambda_a=20)
    elif mode =="test":
        test(net, testloader, ckpt_path,adj)
        #draw_lm(net, testset, ckpt_path, adj)