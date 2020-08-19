from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader

class DeepFashion(Dataset):
    def __init__(self, mode='train'):
        self.root='/data/lxy/Category_and_Attribute_Prediction_Benchmark/'
        self.txtpath=self.root+'hsy_'+mode+'.txt'
        self.transform = transforms.Compose([
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(self.txtpath)
        self.data = f.readlines()
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        line=self.data[idx].split()
        imgpath=self.root+line[0]
        img=Image.open(imgpath)
        x1=int(line[1])
        y1=int(line[2])
        x2=int(line[3])
        y2=int(line[4])
        img=img.crop((x1,y1,x2,y2))
        xx=x2-x1
        yy=y2-y1
        img=img.resize((224, 224))


        landmarks = torch.zeros((8, 4))
        for xxx in range(8):
            for yyy in range(4):
                landmarks[xxx][yyy] = int(line[xxx * 4 + yyy + 5])
            landmarks[xxx][2] = (landmarks[xxx][2] - x1) / float(xx)
            landmarks[xxx][2] = landmarks[xxx][2] * 224.
            landmarks[xxx][3] = (landmarks[xxx][3] - y1) / float(yy)
            landmarks[xxx][3] = landmarks[xxx][3] * 224.

        #category=torch.zeros((50))
        #category[int(line[3*num_landmark+6])]=1.
        category_type=int(line[37])
        category_label=int(line[38])

        label=torch.zeros((1000))
        for i in range(39,len(line)):
            label[int(line[i])]=1.
        sample = {'image': self.transform(img), 'label': label,'landmarks':landmarks,'category':category_label,'category_type':category_type}
        return sample


def gaussian_map(image_w, image_h, center_x, center_y, R):
    Gauss_map = np.zeros((image_h, image_w))

    mask_x = np.matlib.repmat(center_x, image_h, image_w)
    mask_y = np.matlib.repmat(center_y, image_h, image_w)
    x1 = np.arange(image_w)
    x_map = np.matlib.repmat(x1, image_h, 1)
    y1 = np.arange(image_h)
    y_map = np.matlib.repmat(y1, image_w, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map - mask_x)**2 + (y_map - mask_y)**2)
    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    return Gauss_map


def gen_landmark_map(image_w, image_h, landmarks, R):
    ret = []
    for i in range(landmarks.size(0)):
        if landmarks[i][1] == 0:
            ret.append(np.zeros((image_w, image_h)))
        else:
            channel_map = gaussian_map(image_w, image_h, landmarks[i][2], landmarks[i][3], R)
            ret.append(channel_map.reshape((image_w, image_h)))
    return np.stack(ret, axis=0).astype(np.float32)


class DeepFashion_landmark(Dataset):
    def __init__(self, mode='train'):
        self.root='/data/lxy/Category_and_Attribute_Prediction_Benchmark/'
        self.txtpath=self.root+'hsy_'+mode+'.txt'
        self.transform = transforms.Compose([
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(self.txtpath)
        self.data = f.readlines()
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        line=self.data[idx].split()
        imgpath=self.root+line[0]
        img=Image.open(imgpath)
        ini_img=img.copy()
        x1=int(line[1])
        y1=int(line[2])
        x2=int(line[3])
        y2=int(line[4])
        img=img.crop((x1,y1,x2,y2))
        xx=x2-x1
        yy=y2-y1
        img=img.resize((224, 224))


        landmarks = torch.zeros((8, 4))
        for xxx in range(8):
            for yyy in range(4):
                landmarks[xxx][yyy] = int(line[xxx * 4 + yyy + 5])
            landmarks[xxx][2] = (landmarks[xxx][2] - x1) / float(xx)
            landmarks[xxx][2] = landmarks[xxx][2] * 224.
            landmarks[xxx][3] = (landmarks[xxx][3] - y1) / float(yy)
            landmarks[xxx][3] = landmarks[xxx][3] * 224.

        #category=torch.zeros((50))
        #category[int(line[3*num_landmark+6])]=1.
        category_type=int(line[37])
        category_label=int(line[38])

        label=torch.zeros((1000))
        for i in range(39,len(line)):
            label[int(line[i])]=1.

        #sample = {'image': self.transform(img), 'label': label, 'landmarks': landmarks, 'category': category_label,'category_type': category_type}
        sample={}
        sample['image']=self.transform(img)
        sample['label']=label
        sample['landmarks']=landmarks
        sample['category']=category_label
        sample['category_type']=category_type
        sample['x1']=x1
        sample['y1']=y1
        sample['x2'] = x2
        sample['y2'] = y2
        sample['landmark_map'] = torch.from_numpy(gen_landmark_map(224, 224, landmarks, 16))
        sample['img_path']=imgpath
        return sample


class DeepFashion_inshop(Dataset):
    def __init__(self, mode='train'):
        self.root='/data/lxy/In-shop_Clothes_Retrieval_Benchmark/'
        self.txtpath=self.root+'hsy_'+mode+'.txt'
        self.transform = transforms.Compose([
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(self.txtpath)
        self.data = f.readlines()
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        line=self.data[idx].split()
        imgpath=self.root+line[0]
        img=Image.open(imgpath)
        x1=int(line[1])
        y1=int(line[2])
        x2=int(line[3])
        y2=int(line[4])
        img=img.crop((x1,y1,x2,y2))
        xx=x2-x1
        yy=y2-y1
        img=img.resize((224, 224))


        landmarks = torch.zeros((8, 4))
        for xxx in range(8):
            for yyy in range(4):
                landmarks[xxx][yyy] = int(line[xxx * 4 + yyy + 5])
            landmarks[xxx][2] = (landmarks[xxx][2] - x1) / float(xx)
            landmarks[xxx][2] = landmarks[xxx][2] * 224.
            landmarks[xxx][3] = (landmarks[xxx][3] - y1) / float(yy)
            landmarks[xxx][3] = landmarks[xxx][3] * 224.

        #category=torch.zeros((50))
        #category[int(line[3*num_landmark+6])]=1.
        id=int(line[38])
        category_label=int(line[37])-1

        label=torch.zeros((463))
        for i in range(39,len(line)):
            label[int(line[i])]=1.

        #sample = {'image': self.transform(img), 'label': label, 'landmarks': landmarks, 'category': category_label,'category_type': category_type}
        sample={}
        sample['image']=self.transform(img)
        sample['label']=label
        sample['landmarks']=landmarks
        sample['category']=category_label
        sample['id']=id
        sample['x1']=x1
        sample['y1']=y1
        sample['x2'] = x2
        sample['y2'] = y2
        sample['landmark_map'] = torch.from_numpy(gen_landmark_map(224, 224, landmarks, 16))
        sample['img_path']=imgpath
        return sample










if __name__ == '__main__':
    trainset = DeepFashion("train")
    print(trainset[0]['image'].size())
    print(trainset[0]['label'].size())
    print(trainset[0]['landmarks'].size())
    print(trainset[0]['category'].size())
    print(1.0==1)






