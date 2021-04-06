from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from dataset import CXR8_train, CXR8_validation
from model import CX_14

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os


    
net = ResModel()
net.cuda()
num_epochs = 200
gamma = 10
learning_rate = 1e-5
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
loss_function = nn.BCEWithLogitsLoss()

best_auc_ave = 0.0
since = time.time()
best_model_wts = net.state_dict()
best_auc = []
iter_num = 0
model_save_dir = './savedModels'
data_root_dir = './dataset'
class_names = train_loader.dataset.classes
log_dir = './runs'
model_name = 'myNet'

writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name),comment=model_name)
input_tensor = torch.Tensor(1, 3, 512, 512).cuda()
writer.add_graph(net, input_tensor)

for epoch in range(num_epochs):
    running_loss = 0.0
    train_auc = 0.0
    output_list = []
    label_list = []
    loss_list = []
    loss_all = 0.0
    net.train()
    # Iterate over data.
    
    for idx, data in enumerate(train_loader):
        # get the inputs
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()
        
        #calculate weight for loss
        P = 0
        N = 0
        for label in labels:
            for v in label:
                if int(v) == 1: P += 1
                else: N += 1
        if P!=0 and N!=0:
            BP = (P + N)/P
            BN = (P + N)/N
            weights = torch.tensor([BP, BN], dtype=torch.float).cuda()
        #loss_function = nn.CrossEntropyLoss(weights)
        
        optimizer.zero_grad()
        outputs = net(images)

        labels = labels.type_as(outputs)
        loss = loss_function(outputs, labels)
        #print(loss.item())
        loss_all += loss.item()
        #print(loss_all)
        loss.backward()
        optimizer.step()
        iter_num += 1
        
        outputs = outputs.detach().to('cpu').numpy()
        labels = labels.detach().to('cpu').numpy()
        for i in range(outputs.shape[0]):
            output_list.append(outputs[i].tolist())
            label_list.append(labels[i].tolist())
        
        if idx % 10 == 0:
            print("epoch {epoch}, [{trained_samples}/{total_samples}], loss: {:.4f}".format(
                loss.item(),
                epoch=epoch,
                trained_samples=idx * batch_size + len(images),
                total_samples=len(train_loader.dataset)))
            
    writer.add_scalar('train_loss', loss_all/len(train_loader.dataset)*16, epoch)
    try:
        epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
        epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
    except:
        epoch_auc_ave = 0
        epoch_auc = [0 for _ in range(len(class_names))]
        
    writer.add_scalar('train_auc', epoch_auc_ave, epoch)
    log_str = ''
    for i, c in enumerate(class_names):
        log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])
        writer.add_scalar(c + "_train", epoch_auc[i], epoch)
    log_str += '\n'
    print(log_str)
                 
    net.eval()
    val_auc = 0.0
    val_loss = 0.0
    output_list = []
    label_list = []
    
    for idx, data in enumerate(val_loader):
        # get the inputs
        images, labels = data
        
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        
        labels = labels.type_as(outputs)
        with torch.no_grad():
            loss = loss_function(outputs, labels)
        val_loss += loss
        outputs = outputs.detach().to('cpu').numpy()
        labels = labels.detach().to('cpu').numpy()
        for i in range(outputs.shape[0]):
            output_list.append(outputs[i].tolist())
            label_list.append(labels[i].tolist())
        
    try:
        epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
        epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
        print(epoch_auc)
    except:
        epoch_auc_ave = 0
        epoch_auc = [0 for _ in range(len(class_names))]
                  
    log_str = ''
    for i, c in enumerate(class_names):
        log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])
        writer.add_scalar(c + "_val", epoch_auc[i], epoch)
    log_str += '\n'
    print(log_str)
    writer.add_text('log', log_str, epoch)
            
    print("epoch {}, Val loss: {:.4f}, Val AUC {:.4f}".format(epoch,
                                                         val_loss.item() / len(val_loader.dataset),
                                                         val_auc / len(val_loader.dataset)))
    writer.add_scalar('val_loss', val_loss / len(val_loader.dataset), epoch)
    writer.add_scalar('val_auc', epoch_auc_ave, epoch)
    
    if epoch_auc_ave > best_auc_ave:
        best_auc = epoch_auc
        best_auc_ave = epoch_auc_ave
        best_model_wts = net.state_dict()
        model_dir = os.path.join(model_save_dir, str(epoch) + model_name + 'best' +'.pth')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(net.state_dict(), model_dir)
        print('Model saved to %s'%(model_dir))
        
    if epoch % 9 == 0:
        model_dir = os.path.join(model_save_dir, str(epoch) + model_name + 'regular' +'.pth')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(net.state_dict(), model_dir)
        print('Model saved to %s'%(model_dir))
