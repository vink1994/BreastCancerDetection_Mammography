import os
import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler


class AFIMDSMod(torch.utils.data.Dataset):
   
    def __init__(self, samples, transform=None):
        self.num_samples = len(samples)
        self.data = samples
        self.transform = transform

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img, label = self.data[idx]
     
        if self.transform:
            img = self.transform(img)

        return img, label

def AFIMDS_load(root, name, batch_size, num_workers=1, distributed=False, rank=0, world_size=None):
    print("----Loading Breast cancer AFIM_Dataset----")
    TRAIN_TRANSFORM_IMG = torchvision.transforms.Compose([

        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(degrees=(-25, 25))
    ])
    
    VAL_TRANSFORM_IMG = torchvision.transforms.Compose([
        
    ])
    
    
    files = sorted(os.listdir(root))
    
    afim_train_func= torch.load(root+"/"+files[0])
    validation = torch.load(root+"/"+files[1])
    
    train_dataset = AFIMDSMod(afim_train_func, transform=TRAIN_TRANSFORM_IMG)
    eval_dataset = AFIMDSMod(validation, transform=VAL_TRANSFORM_IMG)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

        afim_load_train_module = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler)
        
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=eval_sampler)

    else:
        afim_load_train_module = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print('AFIM_eval_Dataset:', name)
    
    return afim_load_train_module, eval_loader