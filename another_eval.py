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
from dataset import DeepFashion
from torch.nn import DataParallel
#from sync_batchnorm import DataParallelWithCallback as DataParallel
import numpy as np
#from resnet import *
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from model import model_alexnet_v1,model_alexnet_v2,model_vgg16_v2
import pandas as pd
import importlib


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', default="/data/hsy/model_vgg16_v3_2.ckpt", help="path of ckpt file", type=str)
parser.add_argument('--batch_size', default=128, help="batch_size", type=int)


class another_Evaluator(object):

    def __init__(self, category_topk=(1, 3, 5), attr_topk=(3, 5)):
        self.category_topk = category_topk
        self.attr_topk = attr_topk
        self.reset()
        with open('/data/lxy/Category_and_Attribute_Prediction_Benchmark/Anno/list_attr_cloth.txt') as f:
            ret = []
            f.readline()
            f.readline()
            for line in f:
                line = line.split(' ')
                while line[-1].strip().isdigit() is False:
                    line = line[:-1]
                ret.append([
                    ' '.join(line[0:-1]).strip(),
                    int(line[-1])
                ])
        attr_type = pd.DataFrame(ret, columns=['attr_name', 'type'])
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        attr_type.set_index('attr_index', inplace=True)
        self.attr_type = attr_type

    def reset(self):
        self.category_accuracy = []
        self.attr_group_gt = np.array([0.] * 5)
        self.attr_group_tp = np.zeros((5, len(self.attr_topk)))
        self.attr_all_gt = 0
        self.attr_all_tp = np.zeros((len(self.attr_topk),))
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self.category_topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100

            self.category_accuracy.append(res)

    def attr_count(self, output, sample):
        attr_group_gt = np.array([0.] * 5)
        attr_group_tp = np.zeros((5, len(self.attr_topk)))
        attr_all_tp = np.zeros((len(self.attr_topk),))

        target = sample['attr'].cpu().numpy()
        target = np.split(target, target.shape[0])
        target = [x[0, :] for x in target]

        pred = output['attr_output'].cpu().detach().numpy()
        pred = np.split(pred, pred.shape[0])
        pred = [x[0, 1, :] for x in pred]

        for batch_idx in range(len(target)):
            result_df = pd.DataFrame([target[batch_idx], pred[batch_idx]],
                                     index=['target', 'pred'], columns=['attr_' + str(i) for i in range(1000)])
            result_df = result_df.transpose()
            result_df = result_df.join(self.attr_type[['type']])
            ret = []
            for i in range(1, 6):
                ret.append(result_df[result_df['type'] == i]['target'].sum())
            attr_group_gt += np.array(ret)
            ret = []
            result_df = result_df.sort_values('pred', ascending=False)
            attr_all_tp += np.array([
                result_df.head(k)['target'].sum() for k in self.attr_topk
            ])
            for i in range(1, 6):
                sort_df = result_df[result_df['type'] == i]
                ret.append([
                    sort_df.head(k)['target'].sum() for k in self.attr_topk
                ])
            attr_group_tp += np.array(ret)

        self.attr_group_gt += attr_group_gt
        self.attr_group_tp += attr_group_tp

        self.attr_all_gt += attr_group_gt.sum()
        self.attr_all_tp += attr_all_tp

    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * output['lm_pos_output'] - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.attr_count(output, sample)
        #self.landmark_count(output, sample)

    def evaluate(self):
        category_accuracy = np.array(self.category_accuracy).mean(axis=0)
        category_accuracy_topk = {}
        for i, top_n in enumerate(self.category_topk):
            category_accuracy_topk[top_n] = category_accuracy[i]

        attr_group_recall = {}
        attr_recall = {}
        for i, top_n in enumerate(self.attr_topk):
            attr_group_recall[top_n] = self.attr_group_tp[..., i] / self.attr_group_gt
            attr_recall[top_n] = self.attr_all_tp[i] / self.attr_all_gt

        #lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        #lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {
            'category_accuracy_topk': category_accuracy_topk,
            'attr_group_recall': attr_group_recall,
            'attr_recall': attr_recall,#'lm_individual_dist': lm_individual_dist,'lm_dist': lm_dist,
        }





if __name__ == '__main__':
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    batch_size=args.batch_size

    trainset = DeepFashion("train")
    valset=DeepFashion("val")
    testset = DeepFashion("test")
    trainloader = DataLoader(trainset, batch_size,shuffle=True)
    valloader=DataLoader(valset, batch_size)
    testloader = DataLoader(testset, batch_size)

    net = model_vgg16_v2(num_category=50,num_attribute=1000).cuda()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    adj = torch.zeros((8, 8))
    adj[0][2] = 1.
    adj[2][0] = 1.
    adj[1][2] = 1.
    adj[2][1] = 1.
    adj[3][2] = 1.
    adj[2][3] = 1.
    adj[2][5] = 1.
    adj[5][2] = 1.
    adj[5][7] = 1.
    adj[7][5] = 1.
    adj[6][5] = 1.
    adj[5][6] = 1.
    adj[4][5] = 1.
    adj[5][4] = 1.
    adj = adj.cuda()
    # print(net)

    #test===================================================================
    evaluator=another_Evaluator(category_topk=(1, 3, 5), attr_topk=(3, 5))
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()
    t = time.time()
    with torch.no_grad():
        tq = tqdm(testloader, ncols=80, ascii=True)
        for i, batch in enumerate(tq):
            image, label, landmarks, category = batch['image'], batch['label'], batch['landmarks'], batch['category']
            image = image.cuda()
            label = label.cuda()
            landmarks = landmarks.cuda()
            category = category.cuda()

            pre_category, pre_attribute = net(image, landmarks, adj)
            sample={}
            sample['category_label']=category
            sample['attr']=label
            #print('category_label',category.size())
            #print(category.type())
            #print('attr', label.size())
            #print(label.type())


            output={}
            output['category_output']=pre_category
            output['attr_output']=pre_attribute
            #print('category_output',pre_category.size())
            #print(pre_category.type())
            #print('attr_output',pre_attribute.size())
            #print(pre_attribute.type())

            evaluator.add(output, sample)

        ret = evaluator.evaluate()
        for topk, accuracy in ret['category_accuracy_topk'].items():
            print('metrics/category_top{}'.format(topk), accuracy)


        for topk, accuracy in ret['attr_group_recall'].items():
            for attr_type in range(1, 6):
                print('metrics/attr_top{}_type_{}_{}_recall'.format(
                    topk, attr_type, attr_type), accuracy[attr_type - 1]
                )

            print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])







