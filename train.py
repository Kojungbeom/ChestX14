from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torchsummary import summary

from dataset import CXR8_train, CXR8_validation
from model import CX_14, CheXNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

def w_cel(outputs, targets):
    sig_output = torch.sigmoid(outputs)
    pos = 0
    neg = 0
    for target in targets:
        for v in target:
            if int(v) == 1:
                pos += 1
            else:
                neg += 1
    if pos != 0 and neg != 0:
        B_p = (pos + neg) / pos
        B_n = (pos + neg) / neg
        weights = torch.tensor([B_p, B_n], dtype=torch.float).cuda()
    else:
        weights = None
    #loss = -targets * torch.log(sig_output) - (1 - targets) * torch.log(1-sig_output)
    
    if weights is not None:
        loss = -weights[0] * targets * torch.log(sig_output) - weights[1] * (1 - targets) * torch.log(1 - sig_output)
    else:
        loss = -targets * torch.log(sig_output) - (1 - targets) * torch.log(1-sig_output)
    
    return loss.mean()

def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    if args.net == 'CX_14':
        net = CX_14()
    else:
        net = CheXNet()
    net.cuda()
    summary(net, (3,256,256))

    N_CLASSES = 14
    CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    DATA_DIR = './dataset/'
    BATCH_SIZE = 128
    V_BATCH_SIZE = 32

    train_dataset = CXR8_train(
        root_dir=DATA_DIR,
        transform=transforms.Compose([
            transforms.Resize(284),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    valid_dataset = CXR8_validation(
        root_dir=DATA_DIR,
        transform=transforms.Compose([
            transforms.Resize(284),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=V_BATCH_SIZE, shuffle=False)

    learning_rate = 2e-4
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #time of we run the script
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    LOG_DIR = 'runs'
    WEIGHT_DIR = 'checkpoint'
    FULL_LOG_DIR = os.path.join(LOG_DIR, 'CX_14', TIME_NOW)
    FULL_WEIGHT_DIR = os.path.join(WEIGHT_DIR, TIME_NOW)

    if not os.path.exists(FULL_LOG_DIR):
        os.mkdir(FULL_LOG_DIR)
        
    if not os.path.exists(FULL_WEIGHT_DIR):
        os.mkdir(FULL_WEIGHT_DIR)

    writer = SummaryWriter(log_dir=FULL_LOG_DIR)

    for epoch in range(30):  # loop over the dataset multiple times
        print("Epoch:",epoch)
        running_loss = 0.0
        itera = 0
        for idx, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            outputs = net(x)
            loss = w_cel(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            itera += 1
            if idx % 10 == 0:
                print("epoch {epoch}, [{trained_samples}/{total_samples}]".format(
                    loss.item(),
                    epoch=epoch,
                    trained_samples= idx * BATCH_SIZE + len(x),
                    total_samples=len(train_loader.dataset)))

        # ======== validation ======== 
        # switch to evaluate mode
        writer.add_scalar('train_loss', running_loss / idx, epoch)
        net.eval()

        # initialize the ground truth and output tensor
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()

        for idx, (v_x, v_y) in enumerate(valid_loader):
            v_x = v_x.cuda()
            v_y = v_y.cuda()
            gt = torch.cat((gt, v_y), 0)
            output = net(v_x)
            pred = torch.cat((pred, output.data), 0)
    
        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
            writer.add_scalar(CLASS_NAMES[i] + "_val", AUROCs[i], epoch)
        
        net.train()
        # print statistics
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / itera ))
        torch.save(net.state_dict(),
                   os.path.join(WEIGHT_DIR, TIME_NOW, 'Cx14_'+str(epoch + 1)+'_'+str(AUROC_avg)+'.pth'))
               
    print('Finished Training')

