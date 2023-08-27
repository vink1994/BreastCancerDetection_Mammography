import argparse
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.optim as optim
from AFIM_Init_mod import AFIM_Init_mod
from multiprocessing import cpu_count
from afim_train_func_2views import AFIM_Train_Class
from utils.afim_load_data_module import AFIMDS_load
from utils.afim_fetchFile import afim_fetchFile
from arg_parser import get_arguments

print("all the libraries for AFIM model is imported")


def main(rank, world_size, opt):
    lr = opt.lr
    epochs = opt.epochs
    batch_size = opt.batch_size
    AFIM_Dataset = opt.Dataset
    n_workers = opt.n_workers if opt.n_workers != 'max' else cpu_count()

    manualSeed = opt.seed

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)

    model = opt.model
    n = opt.n
    afim_pth_trn = opt.afim_pth_trn
    hyper_param1 = opt.hyper_param1

    if AFIM_Dataset in ['CBIS', 'CBIS_DDSM']:
        num_classes = 1
    elif AFIM_Dataset == 'other':
        num_classes = 5
    else:
        raise RuntimeError('Wrong dataset or not implemented')

    afim_load_train_module, eval_loader = AFIMDS_load(
        root=afim_pth_trn, name=AFIM_Dataset, batch_size=batch_size, num_workers=n_workers, distributed=opt.distributed,
        rank=rank, world_size=world_size)
    AFIM_pretrain_wt= None if opt.afim_mod_eval_hp else opt.AFIM_mod_weight
    net = AFIM_Init_mod(str_model=model, n=n, num_classes=num_classes, weights=AFIM_pretrain_wt, 
                    hyper_param3=opt.hyper_param3, hyper_param2=opt.hyper_param2)  
    
    if opt.afim_mod_eval_hp:
        if AFIM_Dataset != 'CBIS' and hyper_param1 == 2:
            net.add_top_blocks(num_classes=num_classes)
        net.load_state_dict(torch.load(opt.AFIM_mod_weight, map_location='cpu'))
        
    else:
        if opt.hyper_param1 == 2 and opt.AFIM_mod_weight:    
            if opt.hyper_param2:
                print("Loading weights of patch classifier from ", opt.AFIM_mod_weight)
                net = AFIM_Init_mod(str_model=model, n=n, num_classes=5)  
                net.load_state_dict(torch.load(opt.AFIM_mod_weight, map_location='cpu'))
                net.add_top_blocks(num_classes=num_classes)

            else: 
                net.add_top_blocks(num_classes=num_classes)
                print("Loading weights of pretrained whole-image classifier from ", opt.AFIM_mod_weight)
                net.load_state_dict(torch.load(opt.AFIM_mod_weight, map_location='cpu'))
        
    
   
    
    AFIM_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[Proc{rank} Total ]Number of parameters for afim model:', AFIM_params)
    print()
        
    checkpoint_folder = 'checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    if opt.optim == "SGD":
        AFIM_opt_predfnd = optim.SGD(net.parameters(), lr=lr, hyper_param4=opt.hyper_param4, weight_decay=opt.weight_decay)
    if opt.optim == "Adam":
        AFIM_opt_predfnd = optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
    

    afim_opt_train_module = AFIM_Train_Class(net, AFIM_opt_predfnd, epochs=epochs,
                      use_cuda=False,
                      checkpoint_folder=checkpoint_folder,
                      l1_reg=opt.l1_reg,
                      num_classes=num_classes,
                      hyper_param1=hyper_param1,
                      pos_weight=opt.pos_weight,
                      distributed=opt.distributed,
                      rank=rank,
                      world_size=world_size)
    
    if opt.afim_mod_eval_hp:
        afim_opt_train_module.test(eval_loader)
    else:
        afim_opt_train_module.AFIM_train_mod_func(afim_load_train_module, eval_loader)
    
    
if __name__ == '__main__':
    opt=get_arguments()
    main(rank=0, world_size=None, opt=opt)

