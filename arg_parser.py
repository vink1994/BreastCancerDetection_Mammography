import argparse
from utils.afim_fetchFile import afim_fetchFile
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--n_workers', default=1)
    parser.add_argument('--n', type=int, default=2, help="n parameter for AFIM layers")
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--l1_reg', type=bool, default=False)
    parser.add_argument('--afim_pth_trn', type=str, default='/V6/AFIM/Data', help="Folder containg afim_train_funcdata")
    parser.add_argument('--Dataset', type=str, default='CBIS_DDSM', help='CBIS')
    parser.add_argument('--hyper_param1', type=int, default=2, help='setting parameter for deepnet')
    parser.add_argument('--model', type=str, default='AFIMDeepNetmod2', help='Models: ...')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hyper_param4', type=float, default=0.9)
    parser.add_argument('--AFIM_mod_weight', help='model weights for pretraining or testing')
    parser.add_argument('--pos_weight', type=float, help='dir')
    parser.add_argument('--hyper_param3', type=bool, default=True, help='true/false')
    parser.add_argument('--hyper_param2', type=bool, default=True, help='')
    parser.add_argument('--afim_mod_eval_hp', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False, help='True for model')
    parser.add_argument('--AFIM_mod_param_eval', type=str, default='/V6/AFIM/afim_conf_mod/AFIM_config_single.txt', help='Train_func')
    parse_list = afim_fetchFile(parser.parse_args().AFIM_mod_param_eval)
    opt = parser.parse_args(parse_list)
    return opt

