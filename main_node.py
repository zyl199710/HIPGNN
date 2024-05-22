import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, average_precision_score
from sklearn import metrics
from model_node import Specformer
from utils import count_parameters, init_params, seed_everything, get_split, set_random_seed
from sklearn.model_selection import train_test_split

def cos_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=-1)

def normalize(tensor):

    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    if torch.cuda.is_available():
        device_index = args.cuda
        if device_index >= 0 and device_index < torch.cuda.device_count():
            device = f'cuda:{device_index}'
            torch.cuda.set_device(device)
        else:
            print(f"Invalid CUDA device index: {args.cuda}. Falling back to default CUDA device (cuda:0).")
            device = 'cuda:0'
            torch.cuda.set_device(device)
    else:
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    print(device)
    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    hidden_dim1 = config['hidden_dim1']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']
    forward_expansion = config['forward_expansion']

    if args.dataset == "STRINGdb":
        path = '/content/drive/MyDrive/HIPGNN/data/STRINGdb_weighted.pt'
    if args.dataset == "CPDB":
        path = '/content/drive/MyDrive/HIPGNN/data/CPDB_weighted.pt'
    data_loaded = torch.load(path)
    print(path)
    e = data_loaded['e']  # Tensor
    u = data_loaded['u']  # Tensor
    x = data_loaded['x']  # Tensor
    y = data_loaded['y']  # Tensor
    train_mask = data_loaded['train_mask']
    test_mask = data_loaded['test_mask']
    val_mask = data_loaded['val_mask']
    positive_edges = data_loaded['non_zero_edges_with_weights']
    negative_edges = data_loaded['zero_edges_with_weights']


    e_small = e[:args.sm]
    u_small = u[:, :args.sm]
    e_large = e[-args.lm:]
    u_large = u[:, -args.lm:]
    e = torch.cat((e_small, e_large), dim=0)
    u = torch.cat((u_small, u_large), dim=1)

    e, u, x, y = e.cuda(), u.cuda(), x.cuda(), y.cuda()
    x = x.float()
    e = e.float()
    u = u.float()
    print(y.shape)
    print(len(y.size()))
    print(y.size(1))
    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    mask = (train_mask | test_mask | val_mask)
    indices = np.where(mask)[0]
    y_index = y[indices].cpu().numpy()

    n_train_ratio = float(args.n_train)
    print('n_train_ratio')
    print(n_train_ratio)
    e_train_ratio = args.e_train
    print('e_train_ratio')
    print(e_train_ratio)
    train, test = train_test_split(
        indices,
        test_size= 1-n_train_ratio,
        random_state=args.seed,
        stratify=y_index)


    train = torch.tensor(train, dtype=torch.long, device='cuda')
    test = torch.tensor(test, dtype=torch.long, device='cuda')
    train, test = train.cuda(), test.cuda()
    pos_train_edge, pos_test_edge = train_test_split(positive_edges.T, test_size=1- e_train_ratio, random_state=42)
    neg_train_edge, neg_test_edge = train_test_split(negative_edges.T, test_size=1- e_train_ratio, random_state=42)
    print("pos_train_edge shape:", pos_train_edge.shape)
    print("neg_train_edge shape:", neg_train_edge.shape)

    train_edges = np.concatenate([pos_train_edge, neg_train_edge], axis=0)
    test_edges = np.concatenate([pos_test_edge, neg_test_edge], axis=0)
    e_train_labels = torch.cat([
        torch.ones(len(pos_train_edge)),
        torch.zeros(len(neg_train_edge))
    ], dim=0).to(device)
    e_test_labels = torch.cat([
        torch.ones(len(pos_test_edge)),
        torch.zeros(len(neg_test_edge))
    ], dim=0).to(device)

    train_edge_index = train_edges[:, :2].T
    train_edge_weight = train_edges[:, 2]
    train_edge_weight = torch.tensor(train_edge_weight, dtype=torch.float).to(device)

    test_edge_index = test_edges[:, :2].T
    test_edge_weight = test_edges[:, 2]
    test_edge_weight = torch.tensor(test_edge_weight, dtype=torch.float).cpu()

    nfeat = x.size(1)
    net = Specformer(nclass, nfeat, nlayer, hidden_dim, hidden_dim1,num_heads, tran_dropout, feat_dropout, prop_dropout, norm, forward_expansion).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net = net.float()
    print(count_parameters(net))


    best_metrics = {
        'test_acc': 0.0,
        'test_auroc': 0.0,
        'test_f1': 0.0,
        'test_auprc': 0.0,
        'link_pred_auroc': 0.0,
        'link_pred_ap': 0.0,
        'weight_pred_mse': float('inf'),
        'weight_pred_rmse': float('inf'),
        'weight_pred_mae': float('inf')
    }

    best_epoch = -1


    positive_count = y[train].sum()
    negative_count = (y[train] == 0).sum()


    if positive_count > 0 and negative_count > 0:

        pos_weight = negative_count.float() / positive_count.float()
    else:
        pos_weight = torch.tensor(1.0)

    pos_weight = pos_weight.to(device)
    node_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    edge_criterion = torch.nn.BCEWithLogitsLoss()
    if args.dataset == "CPDB":
        lambda_n_cl = 0.02
        lambda_e_pr = (1-lambda_n_cl)/2
        lambda_w_pr = (1 - lambda_n_cl)/2
    if args.dataset == "STRINGdb":
        lambda_n_cl = 0.05
        lambda_e_pr = (1-lambda_n_cl)/2
        lambda_w_pr = (1 - lambda_n_cl)/2

    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()

        n_cl, e_pr, w_pr, c1, c2, c3, c4 = net(e, u, x.to(device))
        n_cl_0 = net.predict_node(n_cl)
        n_cl_f = n_cl_0.float()
        target = y[train].to(device).float()
        n_cl_loss = node_criterion(n_cl_f[train].squeeze(), target)
        src, dest = train_edge_index
        e_pr_f = cos_similarity(e_pr[src], e_pr[dest]).to(device)
        e_pr_loss = edge_criterion(e_pr_f, e_train_labels.float())
        w_pr_f = net.predict_link_weight(w_pr, train_edge_index)
        w_pr_loss = F.mse_loss(w_pr_f, train_edge_weight)

        if args.loss == "auto":
            total_loss =  n_cl_loss/(c1*c1) +  e_pr_loss/(c2*c2) + w_pr_loss/(c3*c3)  + 2 * torch.log(c1 * c2 * c3)
        # total_loss = n_cl_loss / (c1 * c1) + e_pr_loss / (c2 * c2) + w_pr_loss / (c3 * c3) + 2 * torch.log(c1 * c2 * c3)
        # total_loss = lambda_n_cl * n_cl_loss + lambda_e_pr * e_pr_loss + lambda_w_pr * w_pr_loss
        elif args.loss == "average":
            lambda_n_cl = 0.3
            lambda_e_pr = (1 - lambda_n_cl) / 2
            lambda_w_pr = (1 - lambda_n_cl) / 2
            total_loss = lambda_n_cl * n_cl_loss + lambda_e_pr * e_pr_loss + lambda_w_pr * w_pr_loss

        else:
            lambda_n_cl = float(args.loss)
            lambda_e_pr = (1 - lambda_n_cl) /3 * 2
            lambda_w_pr = (1 - lambda_n_cl) /3

            total_loss = lambda_n_cl * n_cl_loss + lambda_e_pr * e_pr_loss + lambda_w_pr * w_pr_loss
        if idx == 1:
            print(lambda_n_cl, lambda_e_pr, lambda_w_pr)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            n_cl, e_pr, w_pr, _, _, _, _ = net(e, u, x.to(device))
            n_cl_0 = net.predict_node(n_cl)
            n_cl_f = n_cl_0.float()
            n_pre = torch.sigmoid(n_cl_f[test])
            n_pre = n_pre.cpu()
            src, dest = test_edge_index
            e_pr_f = cos_similarity(e_pr[src], e_pr[dest]).float()
            e_pre = torch.sigmoid(e_pr_f)
            e_true = e_test_labels.cpu().numpy()
            w_pr_f = net.predict_link_weight(w_pr, test_edge_index).cpu().float()
            w_pr_loss = F.mse_loss(w_pr_f, test_edge_weight.float())
            n_true = y[test].cpu().long()
            n_predictions = (n_pre >= 0.5).float()


            n_auroc = metrics.roc_auc_score(n_true, n_pre)
            precision, recall, _ = metrics.precision_recall_curve(n_true, n_pre)
            n_auprc = metrics.auc(recall, precision)
            n_f1 = metrics.f1_score(n_true, n_predictions, average='macro')
            n_acc = metrics.accuracy_score(n_true, n_predictions)

            e_auroc = metrics.roc_auc_score(e_true, e_pre.cpu().numpy())
            e_precision, e_recall, _ = metrics.precision_recall_curve(e_true, e_pre.cpu().numpy())
            e_auprc = metrics.auc(e_recall, e_precision)
            e_ap = metrics.average_precision_score(e_true, e_pre.cpu().numpy())

            w_pr_rmse = torch.sqrt(w_pr_loss).item()
            w_pr_mae = F.l1_loss(w_pr_f, test_edge_weight.float(), reduction='mean').item()


            print(
                f"Epoch: {idx}, Total Loss: {total_loss.item():.4f}, n_cla loss: {n_cl_loss.item():.4f}, "
                f"e_pre loss: {e_pr_loss.item():.4f}, w_pre loss: {w_pr_loss.item():.4f}, "
                f"Node Classification - ACC: {n_acc:.4f}, AUROC: {n_auroc:.4f}, "
                f"F1 Score: {n_f1:.4f}, AUPRC: {n_auprc:.4f}, Link Prediction - AUROC: {e_auroc:.4f}, "
                f"Average Precision: {e_ap:.4f}, Weight Prediction - MSE: {w_pr_loss.item():.4f}, RMSE: {w_pr_rmse:.4f}, MAE: {w_pr_mae:.4f}"
            )


            if n_auprc > best_metrics['test_auprc']:
                best_metrics['test_acc'] = n_acc
                best_metrics['test_auroc'] = n_auroc
                best_metrics['test_f1'] = n_f1
                best_metrics['test_auprc'] = n_auprc
                best_metrics['link_pred_auroc'] = e_auroc
                best_metrics['link_pred_ap'] = e_ap
                best_metrics['weight_pred_mse'] = w_pr_loss.item()
                best_metrics['weight_pred_rmse'] = w_pr_rmse
                best_metrics['weight_pred_mae'] = w_pr_mae
                best_epoch = idx



            print(f"Best Metrics @ Epoch {best_epoch}: Node Classification - ACC: {best_metrics['test_acc']:.4f}, "
                  f"AUROC: {best_metrics['test_auroc']:.4f}, F1 Score: {best_metrics['test_f1']:.4f}, "
                  f"AUPRC: {best_metrics['test_auprc']:.4f}, Link Prediction - AUROC: {best_metrics['link_pred_auroc']:.4f}, "
                  f"Average Precision: {best_metrics['link_pred_ap']:.4f}, Weight Prediction - MSE: {best_metrics['weight_pred_mse']:.4f}, "
                  f"RMSE: {best_metrics['weight_pred_rmse']:.4f}, MAE: {best_metrics['weight_pred_mae']:.4f}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='STRINGdb')  # 'STRINGdb' or 'CPDB'
    parser.add_argument('--loss', default='0.02') # The number represents the value of alpha, "auto" is Weight learner and "average" is Average weight.
    parser.add_argument('--n_train', default=0.8)
    parser.add_argument('--e_train', default=0.8)  #edge training ratio is fixed as 0.8
    parser.add_argument('--sm', type=int, default=3000)  # 1000, 2000, 3000, 4000
    parser.add_argument('--lm', type=int, default=3000)  # 1000, 2000, 3000, 4000
    import datetime
    args = parser.parse_args()

    now = datetime.datetime.now()

    print("当前日期和时间：", now)

    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    main_worker(args, config)
