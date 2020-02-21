import torch

from model import *
from utils import *
#import torch.nn as nn
#import torch.optim as optim

def optimize(stylenet, img, num_iterations, output_path=None, save_preview_iter=None, print_iter=250):

    def iterate():
        t[0] += 1

        optimizer.zero_grad()
        stylenet(img)
        loss = stylenet.get_loss()
        loss.backward()
        
        if print_iter is not None:
            maybe_print(stylenet, t[0], print_iter, num_iterations, loss)
        if save_preview_iter is not None:
            maybe_save_preview(img, t[0], save_preview_iter, num_iterations, output_path)
        return loss
    
    img = nn.Parameter(img.type(stylenet.dtype))
    optimizer, loopVal = setup_optimizer(img, stylenet.params, num_iterations)
    t = [0]
    while t[0] <= loopVal:
        optimizer.step(iterate)
    
    return img


def optimize2(stylenet, img, num_iterations, output_path=None, save_preview_iter=None, print_iter=250):

    def iterate():
        t[0] += 1

        optimizer.zero_grad()
        stylenet(img)
        loss = stylenet.get_loss()
        loss.backward()
        
        if print_iter is not None:
            maybe_print(stylenet, t[0], print_iter, num_iterations, loss)
        if save_preview_iter is not None:
            maybe_save_preview(img, t[0], save_preview_iter, num_iterations, output_path)
        return loss
    
    img = nn.Parameter(img.type(stylenet.dtype))
    optimizer, loopVal = setup_optimizer(img, stylenet.params, num_iterations)
    t = [0]
    while t[0] <= loopVal:
        optimizer.step(iterate)
    
    return img
