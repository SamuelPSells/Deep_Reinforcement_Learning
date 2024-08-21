"""
Deep Reinforcement Learning - Nerual Network Class
"""

__date__ = "2024-08-09"
__author__ = "Samuel Sells"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------

class DRL_NN(nn.Module):
    def __init__(self,
                neuron_stack,
                ):
        super().__init__()
        self.neuron_stack = neuron_stack

    def evaluate(self,X):
        self.neuron_stack.eval()
        y_hat_tens = self.neuron_stack(
            X
        )
        return y_hat_tens
    
    def forward_only(self,X):
        y_hat_tens = self.neuron_stack(
            X
        )
        return y_hat_tens
    
    def learn(self,
              y_hat,
              y,
              criterion,
              optimizer
              ):
        self.neuron_stack.train()
        loss = criterion(
            y_hat,
            y,
        )
        loss.backward()
        optimizer.step()
        self.zero_grad()