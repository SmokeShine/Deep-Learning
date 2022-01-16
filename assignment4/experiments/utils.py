"""
Helper functions.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)


def train(model, dataloader, optimizer, criterion, scheduler=None):
    model.train()
    
    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data.src.transpose(1, 0)
        target = data.trg.transpose(1, 0)

        translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        # changing boiler plate code to move variable to cuda
        loss = criterion(translation.to(target.device), target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data.src.transpose(1, 0)
            target = data.trg.transpose(1, 0)

            translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)
            # trying to fix cuda issue
            loss = criterion(translation.to(target.device), target)
            total_loss += loss
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9080, -0.5639, -3.5862],
                                  [-1.2683, -0.4294, -2.6910],
                                  [-1.7300, -0.3964, -1.8972],
                                  [-2.3217, -0.4933, -1.2334]]), torch.FloatTensor([[0.9629,  0.9805, -0.5052,  0.8956],
                                                                                    [0.7796,  0.9508, -
                                                                                        0.2961,  0.6516],
                                                                                    [0.1039,  0.8786, -
                                                                                        0.0543,  0.1066],
                                                                                    [-0.6836,  0.7156,  0.1941, -0.5110]])

    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0452,  0.7843, -0.0061,  0.0965],
                                [-0.0206,  0.5646, -0.0246,  0.7761],
                                [-0.0116,  0.3177, -0.0452,  0.9305],
                                [-0.0077,  0.1003,  0.2622,  0.9760]])
        ct = torch.FloatTensor([[-0.2033,  1.2566, -0.0807,  0.1649],
                                [-0.1563,  0.8707, -0.1521,  1.7421],
                                [-0.1158,  0.5195, -0.1344,  2.6109],
                                [-0.0922,  0.1944,  0.4836,  2.8909]])
        return ht, ct

    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031]],
                                          [[-0.4129, -0.1802]],
                                          [[0.0599, -0.0151]],
                                          [[-0.9273, 0.2683]],
                                          [[0.6161, 0.5412]]])
        expected_hidden = torch.FloatTensor([[[0.4912, -0.6078],
                                              [0.4912, -0.6078],
                                              [0.4985, -0.6658],
                                              [0.4932, -0.6242],
                                              [0.4880, -0.7841]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor([[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
                                           -2.6142, -3.1621],
                                          [-2.0260, -2.0121, -3.2508, -3.1249, -2.4581, -1.8520, -2.0798, -1.7596,
                                           -2.6393, -3.2001],
                                          [-2.1078, -2.2130, -3.1951, -2.7392, -2.1194, -1.8174, -2.1087, -2.0006,
                                           -2.4518, -3.2652],
                                          [-2.7016, -1.1364, -3.0247, -2.9801, -2.8750, -3.0020, -1.6711, -2.4177,
                                           -2.3906, -3.2773],
                                          [-2.2018, -1.6935, -3.1234, -2.9987, -2.5178, -2.1728, -1.8997, -1.9418,
                                           -2.4945, -3.1804]])
        expected_hidden = torch.FloatTensor([[[-0.1854, 0.5561],
                                              [-0.4359, 0.1476],
                                              [-0.0992, -0.3700],
                                              [0.9429, 0.8276],
                                              [0.0372, 0.3287]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor([[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
                                            -2.1898],
                                           [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
                                            -2.3045]]])
        return expected_out
