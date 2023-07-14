#导入库
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50
import torch.nn.functional as F
import argparse


#命令行参数
parser = argparse.ArgumentParser()
#type_选1代表既输入文本也输入图像，2代表仅输入图像，3代表仅输入文本
parser.add_argument('--option', type=int, default=1) 
args = parser.parse_args()



#分词+统一文本长度的函数
def text_process(texts,max_length):
    tokenized_texts = [tokenizer(text,padding='max_length',max_length=max_length,truncation=True,return_tensors="pt") for text in texts]
    return tokenized_texts


#创建同时包含图像和文本的数据集的函数
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels,transform=None):
        self.image_paths = image_paths     
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image ,input_ids, attention_mask, labels

    
#特征提取模型
#图像特征提取模型
class img_feature(nn.Module):
    def __init__(self):
        super(img_feature, self).__init__()
        self.resnet = resnet50(pretrained=True) #使用resnet50提取图像特征
    
    def forward(self, image):
        features = self.resnet(image)
        return features

#文本特征提取模型
class text_feature(nn.Module):
    def __init__(self):
        super(text_feature, self).__init__()
        self.bert = pretrained_model #使用Bert提取文本特征

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = pooled_output
        return output


#多模态融合模型
class multi_fusion_model(nn.Module):
    def __init__(self, num_classes, type_):
        super(multi_fusion_model, self).__init__()
        self.image_feature = img_feature()
        self.text_feature = text_feature()
        self.type_ = type_
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, image, input_ids, attention_mask):
        #图像+文本
        if(self.type_ == 1):
            image_features = self.image_feature(image)
            text_features = self.text_feature(input_ids,attention_mask)
            fusion_features = torch.cat((text_features,image_features), dim=-1)
            output = self.classifier1(fusion_features)
        #仅图像
        elif(self.type_ == 2):
            image_features = self.image_feature(image)
            output = image_features
            output = self.classifier2(image_features)
        #仅文本
        else:
            text_features = self.text_feature(input_ids,attention_mask)
            output = self.classifier3(text_features)
        return output
    

#标签预测函数
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask, _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions




if __name__ == "__main__":
    #加载BERT模型和分词
    pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    #将图像缩放到统一大小并转换为tensor
    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(),])
    
    
    #数据读取与预处理
    #数据文件路径
    folder_path = './data/'
    #读取用于训练的标签文件
    train_label = pd.read_csv('train.txt',sep=",")
    #将标签文件中的情感标签对应替换为0、1、2方便后续训练
    column_dict = {"positive": 0, "negative": 1,"neutral":2}
    new_train_label = train_label.replace({"tag": column_dict})
    labels = list(new_train_label['tag'])

    #获取所有图像的路径
    image_paths = []
    for seq_num in new_train_label['guid']:
        image_path = folder_path+str(seq_num)+'.jpg'
        image = cv2.imread(image_path)
        image_paths.append(image_path)
    #print(len(image_paths))

    #获取所有文本
    texts=[]
    for seq_num in new_train_label['guid']:
        path = folder_path+str(seq_num)+'.txt'
        with open(path, "r",encoding='gb18030') as file:
            content = file.read()
            texts.append(content)
    #print(len(texts))

    #将训练数据划分为训练集与验证集（8：2）
    image_paths_train, image_paths_valid, texts_train, texts_valid, labels_train, labels_valid = train_test_split(image_paths, texts, labels, test_size=0.2, random_state=2023)

    #最大文本长度限制
    max_length = 131  
    #文本预处理（分词+统一长度）
    tokenized_texts_train = text_process(texts_train, max_length)
    tokenized_texts_valid = text_process(texts_valid, max_length)

    #合并图像与文本，创建数据集
    dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
    dataset_valid = Dataset(image_paths_valid, tokenized_texts_valid, labels_valid, transform)
    
    
    #正式训练
    #参数设置
    num_classes=1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    lr = 1e-6
    num_epochs = 3
    type_ = 1
    model = multi_fusion_model(num_classes, type_)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    best_model = None

    #数据加载器
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    
    #模型训练
    for epoch in range(num_epochs):
        model.train()  
        train_loss = 0
        train_correct = 0 
        for images, input_ids, attention_mask, labels in loader_train:
            images = images.to(device)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)     
            labels = labels.to(device)     
            optimizer.zero_grad()     
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            train_loss += loss.item()
        tr_loss = train_loss / len(loader_train)
        tr_acc = train_correct.item() / len(loader_train.dataset) 

        predictions = predict(model, loader_valid, device)

        val_predictions = np.array(predictions)
        val_labels = np.array(labels_valid)
        val_acc = (val_predictions == val_labels).sum() / len(val_labels)
        if(val_acc > best_acc):
            best_acc = val_acc
            best_model = model
            torch.save(model, 'best_model.pt')
        print(f"batch_size: {batch_size}, lr: {lr}, Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Val Acc: {val_acc:.4f}")
    print(f"best_acc: {best_acc}")
    
    
    #读取测试数据，生成并保存情感预测标签
    test_data = pd.read_csv("test_without_label.txt",sep=",")
    test_labels = np.array(test_data['tag'])

    image_paths_test = []
    for seq_num in test_data['guid']:
        image_path = folder_path+str(seq_num)+".jpg"
        image = cv2.imread(image_path)
        image_paths_test.append(image_path)

    texts_test = []
    for seq_num in test_data['guid']:
        path = folder_path+str(seq_num)+".txt"
        with open(path, "r",encoding='gb18030') as file:
            content = file.read()
            texts_test.append(content)

    tokenized_texts_test = text_process(texts_test, max_length)
    dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    model = torch.load('best_model.pt').to(device)
    test_predictions = np.array(predict(model, loader_test, device)) 
    column_dict_diverse = {0:"positive", 1:"negative",2:"neutral"}
    test_data["tag"] = test_predictions
    predictions = test_data.replace({"tag": column_dict_})
    predictions.to_csv('test_without_label.txt',sep=',',index=False)
