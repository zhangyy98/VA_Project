#!/usr/bin/env python
import os
import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from tools.dataset import load_train_dataset, load_test_dataset
from tools import utils
import evaluate

from tensorboardX import SummaryWriter


def train(train_loader, model, criterion, optimizer, epoch, opt, logger=None):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('float32')
        target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2)

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target

        # put the data into Variable
        vfeat_var = Variable(vfeat0).cuda()
        afeat_var = Variable(afeat0).cuda()
        target_var = Variable(target).cuda()

        # forward, backward optimize
        sim = model(vfeat_var, afeat_var)  # inference similarity
        loss = criterion(sim, target_var)

        # update loss in the loss meter
        losses.update(loss.data[0], vfeat0.size(0))

        # compute gradient and do sgd
        optimizer.zero_grad()
        loss.backward()

        # update parameters
        optimizer.step()

        # logger=None means no logger
        if logger:
            logger.add_scalar('loss', loss.data[0], epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

    return losses.avg


# main function for training the model
def main(opt):
    train_loader = load_train_dataset(opt)
    test_loader = load_test_dataset(opt)

    # create model
    print('shift model and criterion to GPU .. ')
    model = models.ResidualRNNConv().cuda()
    print(model)
    criterion = nn.BCELoss().cuda()

    # log directory
    logger = SummaryWriter(comment=f'_{model.__class__.__name__}')

    if opt.init_model:
        print(f'loading pretrained model from {opt.init_model}')
        model.load_state_dict(torch.load(opt.init_model, map_location=lambda storage, loc: storage.cuda()))

    # optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=opt.scheduler_patience)

    accuracy = []
    for epoch in range(opt.max_epochs):
        loss = train(train_loader, model, criterion, optimizer, epoch + 1, opt, logger)
        scheduler.step(loss)

        right, simmat = evaluate.test(test_loader, model, opt)
        accuracy.append(right)  # test accuracy
        logger.add_scalar('accuracy', accuracy[-1], epoch + 1)
        logger.add_histogram('simmat', simmat, epoch + 1, 'auto')

        if right > 0.8:
            path_checkpoint = os.path.join(opt.checkpoint_folder,
                                           f'{model.__class__.__name__}_{epoch + 1}_{right:.2%}.pth')
            utils.save_checkpoint(model.state_dict(), path_checkpoint)
    print(f'Max test accuracy: {np.max(accuracy):.2%} at epoch {(np.argmax(accuracy)+1)}')


if __name__ == '__main__':
    opt = utils.config('./configs/config.yaml')
    main(opt)
