import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

# Dataset 實作
class Dataset():
    def __init__(self, image_dir = 'iclevr', mode = 'train'):
        self.mode = mode
        with open(f"{mode}.json", 'r') as f:
            self.Json = json.load(f)
        
        with open(f'objects.json') as f:
            self.Json_obj = json.load(f)
        
        self.all_objects = list(self.Json_obj.keys())
        self.num_classes = len(self.all_objects)
        self.obj_dic = {obj: id for id, obj in enumerate(self.all_objects)}

        self.image_dir = image_dir
        self.image_list = list(self.Json.keys()) if mode == 'train' else []
        Json_encode = self.Json.values() if mode == 'train' else self.Json
        self.label_list = [self.Encoder(objs) for objs in Json_encode]

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.Json)

    def __getitem__(self, id):
        label = self.label_list[id]
        if self.mode != 'train':
            return label
        
        image_name = self.image_list[id]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label
    
    def Encoder(self, labels):
        vec = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.obj_dic:
                vec[self.obj_dic[label]] = 1
        return vec
            

# 測試載入
if __name__ == '__main__':
    
    dataset = Dataset(mode='train')
    print(len(dataset))
    print(dataset[0][0].shape) # image
    print(dataset[0][1]) # label
    
    dataset = Dataset(mode='test')
    print(len(dataset))
    print(dataset[0])