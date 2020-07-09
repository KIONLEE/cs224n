#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, e_word):
        """ Init CNN.

        @param e_word (int): Output embedding size of target char.
        @param e_word (int): Output embedding size of target word.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=5, padding=1)
        self.ReLU = nn.ReLU() # 'inplace' default: False
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x_reshaped): # x_reshaped => x_conv_out
        """ Forward pass of CNN.

        @param x_reshaped (Tensor): tensor after padding and embedding lookup, shape (src_len * batch_size, e_char, m_word)
        
        @returns x_conv_out (Tensor): output tensor after highway layer, shape (src_len * batch_size, e_word)
        """
        relu = self.ReLU(self.conv(x_reshaped)) # (src_len * b, e_word, m_word - kernel_size + 1)
        x_conv_out = self.maxpool(relu).squeeze(-1) # (src_len * b, e_word)

        return x_conv_out

    ### END YOUR CODE

