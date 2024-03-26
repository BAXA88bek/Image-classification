import torch, pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from glob import glob
from torchvision import transforms as T

class Cloud(Dataset):
    def __init__(self, root, transformations = None):
        
        self.transformations = transformations
      
        self.classes, self.im_ids ,self.class_name , count = {}, [],[], 0
        
        
        ds = pd.read_csv(f"{root}/cloud_classification_export.csv")
        for ind in range(len(ds)):
            
            image_id = f"{root}/{ds.iloc[ind]['image']}"; class_names = ds.iloc[ind]["choice"]
            self.im_ids.append(image_id); self.class_name.append(class_names)
            
            if class_names not in self.classes: self.classes[class_names] = count; count+=1


    def __len__(self): return len(self.im_ids)

    def __getitem__(self, idx):
        im = Image.open(self.im_ids[idx]).convert("RGB")
        gt = self.classes[self.class_name[idx]]

        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt

def get_dls(root, transformations, bs, split = [0.9, 0.05, 0.05]):
    
    ds = Cloud(root = root, transformations = transformations)
    
    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = int(total_len * split[1])
    ts_len = total_len - (tr_len + vl_len)
    
    tr_ds, vl_ds, ts_ds = random_split(dataset = ds, lengths = [tr_len, vl_len, ts_len])
    
    tr_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = 0)
    val_dl = DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = 0)
    ts_dl   = DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = 0)
    
    return tr_dl, val_dl, ts_dl, ds.classes

root = "data/cloud_classification"
