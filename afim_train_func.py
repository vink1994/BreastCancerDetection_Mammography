import torch
import time
import torch.nn as nn
import sys
from AFIM_My_model.afim_mod_layer import AFIMDeepConv
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('V6/AFIM/init_lib/')
print (sys.path.append)
from pytorchtools import EarlyStopping
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
class AFIM_Train_Class():
    def __init__(self, net, AFIM_opt_predfnd, epochs,
                      use_cuda=False, gpu_num=0,
                      checkpoint_folder="./checkpoints",
                      l1_reg=False,
                      num_classes=1,
                      hyper_param1=1,
                      pos_weight=None,
                      distributed=False,
                      rank=0,
                      world_size=None):

        self.AFIM_opt_predfnd = AFIM_opt_predfnd
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoints_folder = checkpoint_folder
        self.l1_reg = l1_reg
        self.rank = rank
        self.distributed = distributed
        self.world_size = world_size
        self.num_classes = num_classes
        self.hyper_param1 = hyper_param1

        if num_classes == 1:
            pos_weight = torch.tensor([pos_weight]) if pos_weight else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.val_criterion = nn.BCEWithLogitsLoss()

        else:
            self.criterion = nn.CrossEntropyLoss()
            self.val_criterion = self.criterion
        
        self.net = net

    def AFIM_train_mod_func(self, afim_load_train_module, eval_loader):
        
        early_stopping = EarlyStopping(patience=20, path=self.checkpoints_folder + "/best_" + run_name + ".pt", rank=self.rank)
        
        for epoch in range(self.epochs):  
            
            if self.distributed:
                afim_load_train_module.sampler.set_epoch(epoch)  

            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            total = 0.0
            correct = 0.0
            y_pred = torch.empty(0)
            y_true = torch.empty(0)
           
            for i, data in enumerate(afim_load_train_module, 0):
                inputs, labels = data

                if self.hyper_param1 == 4:
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                    
                self.AFIM_opt_predfnd.zero_grad()
                
                if self.hyper_param1 == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                if self.l1_reg:
                    
                    regularization_loss = 0.0
                    for child in self.net.children():
                        for layer in child.modules():
                            if isinstance(layer, AFIMDeepConv):
                                for param in layer.a:
                                    regularization_loss += torch.sum(abs(param))
                    loss += 0.001 * regularization_loss


                loss.backward()
                self.AFIM_opt_predfnd.step()

                running_loss_train += loss.item()
                
            end = time.time()
            
           
            self.net.eval()
            
            if self.distributed:
                to_gather = dict(y_pred=None, y_true=None, loss_eval=None, loss_train=running_loss_train)
                
            for j, eval_data in enumerate(eval_loader, 0):
                inputs, labels = eval_data

                if self.hyper_param1 == 4:
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                    
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                
                if self.hyper_param1 == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)
                    
                eval_outputs = self.net(inputs)
                eval_loss = self.val_criterion(eval_outputs, labels)
                running_loss_eval += eval_loss.item()

                # for multi-class (patch)
                if self.num_classes == 1:
                    predicted = torch.sigmoid(eval_outputs) > 0.5
                else:
                    _, predicted = torch.max(eval_outputs.data, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100*correct/total

                y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

            
            if self.distributed:
                to_gather["y_pred"] = y_pred
                to_gather["y_true"] = y_true
                to_gather["loss_eval"] = running_loss_eval
                gathered = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered, to_gather)
                y_pred = torch.cat((gathered[0]["y_pred"], gathered[1]["y_pred"]))
                y_true = torch.cat((gathered[0]["y_true"], gathered[1]["y_true"]))
                total = y_truee.shape[0]
                correct = (y_predd == y_true).sum().item()
                acc = 100*correct/total

                if self.num_classes == 1:
                    auc = roc_auc_score(y_truee, y_predd)
                running_loss_train = gathered[0]["loss_train"] + gathered[1]["loss_train"]
                running_loss_eval = gathered[0]["loss_eval"] + gathered[1]["loss_eval"]
                
                i *= 2
                j *= 2
                
            elif self.num_classes == 1:
                auc = roc_auc_score(y_truee, y_truee)
           
            if self.num_classes == 1:
                early_stopping(auc, self.net)
            else:
                early_stopping(acc, self.net)

            if early_stopping.early_stop:
                
                break
            
            running_loss_train = 0.0
            running_loss_eval = 0.0
            self.net.AFIM_train_mod_func()
            

        
            
    def test(self, test_loader):
        print("Evaluating the model for metrics on test AFIM_Dataset")
        
        for name, AFIM_params in self.net.named_parameters():
            AFIM_params.requires_grad = False

        self.net.eval()

        correct = 0.0
        total = 0.0
        y_pred = torch.empty(0)
        y_true = torch.empty(0)
        
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data    

                if self.hyper_param1 == 4:              
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                    
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)
   
                if self.hyper_param1 == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)
                
                eval_outputs = self.net(inputs)

                if self.num_classes == 1:
                    predicted = torch.sigmoid(eval_outputs) > 0.5
                else: 
                    _, predicted = torch.max(eval_outputs.data, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))
                y_preds = torch.load('/AFIM/weights/y_preds.pt')
                y_trues = torch.load('/AFIM/weights/y_trues.pt')
   
        if self.num_classes == 1:
            auc = roc_auc_score(y_trues, y_preds)
            print('AUC %s on CBIS dataset for single views is : %.3f' % (self.net.__class__.__name__, auc))
            
        
 
       
        
