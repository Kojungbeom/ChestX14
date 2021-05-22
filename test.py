from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torchsummary import summary

from dataset import CXR8_validation
from model import CX_14, CheXNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

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
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    if args.net == 'CX_14':
        net = CX_14()
    else:
        net = CheXNet()
    net.cuda()
    net.load_state_dict(torch.load(args.weights))
    summary(net, (3,256,256))

    N_CLASSES = 14
    CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    DATA_DIR = './dataset/'
    BATCH_SIZE = 128
    T_BATCH_SIZE = 32

    test_dataset = CXR8_validation(
        root_dir=DATA_DIR,
        transform=transforms.Compose([
            transforms.Resize(284),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=T_BATCH_SIZE, shuffle=False)

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for idx, (t_x, t_y) in enumerate(test_loader):
        t_x = t_x.cuda()
        t_y = t_y.cuda()
        gt = torch.cat((gt, t_y), 0)
        output = net(t_x)
        pred = torch.cat((pred, output.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))



