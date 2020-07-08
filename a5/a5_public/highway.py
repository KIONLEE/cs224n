#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        """ Init Highway.

        @param e_word (int): Output embedding size of target word.
        """
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(e_word, e_word)
        self.gate_layer = nn.Linear(e_word, e_word)
        self.ReLU = nn.ReLU() # 'inplace' default: False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out): # x_conv_out => x_highway
        """ Forward pass of Highway.

        @param x_conv_out (Tensor): tensor from convolutional layer, shape (e_word,)
        
        @returns x_highway (Tensor): output tensor after highway layer, shape (e_word,)
        """
        x_proj = self.ReLU(self.proj_layer(x_conv_out)) # (e_word,)
        x_gate = self.sigmoid(self.gate_layer(x_conv_out)) # (e_word,)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        return x_highway

    ### END YOUR CODE

