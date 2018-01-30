#!/usr/bin/env python
import numpy as np

import torch
from torch.autograd import Variable

import models
from tools.dataset import load_test_dataset
from tools import utils


def test(test_loader, model, opt):
    """
    test on the test set
    """
    # eval mode
    model.eval()

    simmat = []
    for vfeat, afeat, _ in test_loader:
        vfeat_var = Variable(vfeat, volatile=True).cuda()
        afeat_var = Variable(afeat, volatile=True).cuda()

        cur_sim = model(vfeat_var, afeat_var)
        simmat.append(cur_sim.data.cpu())
    feat_n = int(len(test_loader.dataset) ** 0.5)
    simmat = torch.cat(simmat).resize_(feat_n, feat_n).transpose(0, 1).numpy()
    # return the index of the elements in ascending order along each column
    indices = np.argsort(simmat, 0)
    # get the index matrix of first five row with all columns
    topk = indices[:opt.topk, :]
    right = np.count_nonzero(topk == np.arange(feat_n)) / feat_n
    print(f'Testing accuracy (top{opt.topk}): {right:.2%}')
    return right, simmat


def main(opt):
    test_loader = load_test_dataset(opt)

    # create model
    print('shift model to GPU .. ')
    model = models.ResidualRNNConv().cuda()

    assert opt.init_model
    print(f'loading pretrained model from {opt.init_model}')
    model.load_state_dict(torch.load(opt.init_model, map_location=lambda storage, loc: storage.cuda()))

    test(test_loader, model, opt)


if __name__ == '__main__':
    opt = utils.config('./configs/config.yaml')
    main(opt)
