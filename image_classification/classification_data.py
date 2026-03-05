__all__ = ["ImageDataset"]
from PIL import Image
import os
import pandas as pd
from  torch.utils.data import Dataset
import classification_config
import re
'''

'''
def sorted_alphanumeric(data):
    convert = lambda  text:int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key:[convert(c) for  c in re.split('([0-9]+)', key)]
    return sorted(data,key=alphanum_key)

class ImageDataset:
    def __init__(self,main_dir,fashion_path,transform = None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))
        self.classifications = pd.read_csv(classification_config.FASHION_LABELS_PATH)
        self.label_dict = dict(zip(self.classifications['id'],self.classifications['target']))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir,self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        img_flag_id = self.label_dict[idx]
        if self.transform is not  None:
            tensor_image = self.transform(image)
        else:
            raise RuntimeError("transform参数不能为None，需指定预处理方法")
        return tensor_image,img_flag_id




