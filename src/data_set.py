from torch.utils.data import Dataset
import logging
import os
from PIL import Image, ImageDraw
import json
import ast
import numpy as np

logger = logging.getLogger(__name__)
WORKING_PATH="/root/autodl-tmp/dataset/pheme"
# IMAGE_PATH="/root/autodl-tmp/dataset/weibo/images/"
# IMAGE_PATH="/root/autodl-tmp/dataset/weibo21/"
IMAGE_PATH="/root/autodl-tmp/dataset/pheme/pheme_image/pheme_image/images/"
# IMAGE_PATH="/root/autodl-tmp/dataset/"



def mask_image_with_predefined_patches(image, patch_size, mask_indices, mask_color=(0, 0, 0)):
    """
    Index occluded images according to predefined occlusion patches (optimized version, accelerated with numpy).

    param Image: PIL.Image object, loaded via image.open.
    param patch_size: The size of each patch (assumed to be square).
    param mask_indices: List of predefined occlusion patch indices
    param mask_color: The occlusion color; default is black (0, 0, 0).
    return: The occluded image.
    """

    width, height = image.size

    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(f"图像尺寸 ({width}, {height}) 不是 patch 大小 {patch_size} 的整数倍，请调整图像或 patch 大小。")

    # The number of transverse and longitudinal patches is calculated
    num_patches_x = width // patch_size

    # Convert PIL image to numpy array
    image_array = np.array(image)

    # Make sure the occlusion color is a numpy array
    mask_color = np.array(mask_color, dtype=image_array.dtype)

    # The selected patch is masked
    for index in mask_indices:
        # The position of the patch is calculated based on the index
        patch_x = (index % num_patches_x) * patch_size
        patch_y = (index // num_patches_x) * patch_size

        # Batch occlusion of this patch area
        image_array[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :] = mask_color

    # Convert the numpy array back to a PIL.Image object
    masked_image = Image.fromarray(image_array)

    return masked_image

class MyDataset(Dataset):
    def __init__(self, mode, text_name, text_mask_rate, image_mask_rate):
        self.text_name = text_name
        self.data = self.load_data(mode, text_mask_rate, image_mask_rate)
        self.image_ids=list(self.data.keys())
        self.text_mask_rate = text_mask_rate
        self.image_mask_rate = image_mask_rate

            
    
    def load_data(self, mode, text_mask_rate, image_mask_rate):
        cnt = 0
        data_set=dict()
        
        text_mask = "text_mask_"+str(text_mask_rate)
        image_mask = "image_mask_"+str(image_mask_rate)
        
        if mode in ["train"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            print(mode,"data size", len(datas))
            for i, data in enumerate(datas):
                image = data['image']
                sentence = data[text_mask]
#                 print(sentence)
                label =data['label']
                #print(image, sentence, label)
                # print(IMAGE_PATH+image)
                data['id'] = i
                if os.path.isfile(IMAGE_PATH+image):
                    data_set[data['id']]={"image_path":IMAGE_PATH+image, "text":sentence, 'label': label, "image_mask_indices":ast.literal_eval(data[image_mask])}
                    cnt += 1
                else:
                    print(IMAGE_PATH+image, "is not a file")
                    
        
        if mode in ["test"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            print(mode,"data size", len(datas))
            for i, data in enumerate(datas):
                image = data['image']
                sentence = data[text_mask]
#                 print(sentence)
                label =data['label']
                #print(image, sentence, label)
                data['id'] = i+100000
                if os.path.isfile(IMAGE_PATH+image):
                    data_set[data['id']]={"image_path":IMAGE_PATH+image, "text":sentence, 'label': label, "image_mask_indices":ast.literal_eval(data[image_mask])}
                    cnt += 1
                else:
                    print(IMAGE_PATH+image, "is not a file")
                    
        if mode in ["val"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            print(mode,"data size", len(datas))
            for i, data in enumerate(datas):
                image = data['image']
                sentence = data[text_mask]
#                 print(sentence)
                label =data['label']
                #print(image, sentence, label)

                data['id'] = i+1000000
                if os.path.isfile(IMAGE_PATH+image):
                    data_set[data['id']]={"image_path":IMAGE_PATH+image, "text":sentence, 'label': label, "image_mask_indices":ast.literal_eval(data[image_mask])}
                    cnt += 1
                else:
                    print(IMAGE_PATH+image, "is not a file")
                    
                    
        print(mode, " dataset valid num", cnt)
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["text"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        image_feature = image_feature.convert("RGB")
        image_feature = image_feature.resize((224, 224))
        masked_image = mask_image_with_predefined_patches(image_feature, patch_size=32, mask_indices=self.data[id]["image_mask_indices"], mask_color=(0, 0, 0))
        
        label = self.data[id]["label"]
        return text,masked_image, label, id

    def __len__(self):
        return len(self.image_ids)
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list

