# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import time
import setproctitle
import logging

from dataset_processing import create_dataset, split_train_val_test, make_batch
from model import MSC
from utils import args_printer

torch.autograd.set_detect_anomaly(True)


def setup_logging(args):
    with open(args.log_file, 'a') as file:
        pass
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def init_model(args):
    # model
    if args.model == 'msc':
        model = MSC(args).to(args.device)
    else:
        raise NotImplementedError(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))

    return (model, optimizer)


def train(data, train_loaders, valid_id):
    model.train()
    start = time.time()
    min_loss = 1e5
    patience = 0
    data = data.to(args.device)
    for epoch in range(args.epochs):
        print('Epoch {}:'.format(epoch))
        mae_loss = 0.
        num_iters = len(train_loaders)
        for batch_idx, train_ids in enumerate(train_loaders):
            optimizer.zero_grad()
            out = model(data)
            loss = F.l1_loss(out[train_ids], data.y[train_ids], reduction='sum')
            mae_loss += F.l1_loss(out[train_ids], data.y[train_ids], reduction='sum').item()/num_train
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
        # mae_loss /= num_train
        print("FOLD {}, Time {:.4f} -- Training loss:{}".format(fold, time_iter, mae_loss))
        val_loss, _, _ = test(model, data, valid_id)
        print("FOLD {}, Time {:.4f} -- Validation loss:{}".format(fold, time_iter, val_loss))
        if val_loss < min_loss:
            torch.save(model.state_dict(),
                       '../model/model_after_finetune-{}-{}-{}-{}-{}-{}'.format(args.model, args.lr,
                                                                                args.weight_decay,
                                                                                args.n_hidden, args.batch_size,
                                                                                args.n_embedding))
            print("!!!!!!!!!! Model Saved !!!!!!!!!!")
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break


def test(model, data, test_ids):
    model.eval()
    data = data.to(args.device)
    out = model(data)
    mae = F.l1_loss(out[test_ids], data.y[test_ids], reduction='mean')
    mse = F.mse_loss(out[test_ids], data.y[test_ids], reduction='mean')
    rmse = torch.sqrt(mse)
    nrmse = rmse/(torch.max(data.y[test_ids]) - torch.min(data.y[test_ids]))
    return mae.item(), rmse.item(), nrmse.item()


if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Graph convolutional networks for influencer value prediction')
    parser.add_argument('-sd', '--seed', type=int, default=630, help='random seed')
    parser.add_argument('-lr', '--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=125, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('-d', '--dropout_ratio', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-dvs', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str, default='gcn', help='model')
    parser.add_argument('-dp', '--dataset_path', type=str, default='../data/sample2_dataset_norm.npy',
                        help='node feature matrix data path')
    parser.add_argument('-nh', '--n_hidden', type=int, default=32, help='number of hidden nodes in each layer of GCN')
    parser.add_argument('-p', '--patience', type=int, default=150, help='Patience')
    parser.add_argument('--n_embedding', type=int, default=20, help='embedding size')
    parser.add_argument('--n_folds', type=int, default=10, help='n_folds')
    parser.add_argument('--log_file', type=str, default='../log/test.log', help='log file')
    parser.add_argument('-sr', '--seed_ratio', type=float, default=0.2, help='seed ratio')
    args = parser.parse_args()
    args_printer(args)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    setproctitle.setproctitle('msc@zhangguozhen')

    logger = setup_logging(args)
    start_time = time.time()

    print('------------------------- Loading data -------------------------')
    dataset = create_dataset(os.path.join('..', 'data', args.dataset_path), args)
    args.num_features = dataset.num_features
    args.num_edge_features = dataset.num_edge_features
    args.num_communities = int(dataset.community.max().item() + 1)
    args.num_nodes = dataset.x.size(0)
    train_ids, val_ids, test_ids = split_train_val_test(args.num_communities, args.n_folds, args.seed)

    mae_folds = []
    rmse_folds = []
    nrmse_folds = []
    for fold in range(args.n_folds):
        train_loaders, num_train = make_batch(train_ids[fold], args.batch_size, args.seed)

        print('\nFOLD {}, train {}, valid {}, test {}'.format(fold, num_train, len(val_ids[fold]), len(test_ids[fold])))

        print('\n--------------- Initialize Model ---------------')
        model, optimizer = init_model(args)

        print('\n--------------- Training ---------------')
        train(dataset, train_loaders, val_ids[fold])

        print('\n--------------- Testing ---------------')
        model.load_state_dict(
            torch.load('../model/model_after_finetune-{}-{}-{}-{}-{}-{}'.format(
                args.model, args.lr, args.weight_decay,
                args.n_hidden, args.batch_size,
                args.n_embedding)))
        mae_loss, rmse_loss, nrmse_loss = test(model, dataset, test_ids[fold])
        mae_folds.append(mae_loss)
        rmse_folds.append(rmse_loss)
        nrmse_folds.append(nrmse_loss)

        print('---------------------------------------')
        print('mae_loss: {}'.format(mae_loss))

    logger.info('model:%s, n_hidden:%d, lr:%f, weight_decay:%f, batch_size:%d, n_embedding:%d',
                args.model, args.n_hidden, args.lr, args.weight_decay, args.batch_size, args.n_embedding)
    logger.info('mae_folds: %s', str(mae_folds))
    logger.info('%d-fold cross validation avg mae (+- std): %f (%f)', args.n_folds, np.mean(mae_folds), np.std(mae_folds))
    logger.info('%d-fold cross validation avg rmse (+- std): %f (%f)', args.n_folds, np.mean(rmse_folds), np.std(rmse_folds))
    logger.info('%d-fold cross validation avg nrmse (+- std): %f (%f)', args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds))
    logger.info('---------------------------------------------------------')

    args_printer(args)
    print('Total train time: {}', time.time()-start_time)
    print('{}-fold cross validation avg mae (+- std): {} ({})'.format(args.n_folds, np.mean(mae_folds), np.std(mae_folds)))
    print('{}-fold cross validation avg rmse (+- std): {} ({})'.format(args.n_folds, np.mean(rmse_folds), np.std(rmse_folds)))
    print('{}-fold cross validation avg nrmse (+- std): {} ({})'.format(args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds)))
    mae_folds = ['{:.2f}'.format(u) for u in mae_folds]
    mae_folds = [float(u) for u in mae_folds]
    print(mae_folds)
