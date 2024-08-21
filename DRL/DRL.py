"""
Enter script name

Enter short description of the script
"""

__date__ = "2024-08-14"
__author__ = "Samuel Sells"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
sys.path.append(r'./DRL_sub_classes')
from DRL_sub_classes import DRL_NN

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------

class DeepReinforcementLearner(nn.Module):
    def __init__(
        self,
        observation_max = tuple, # starting at 0 for each dimension
        action_max = tuple, # starting at 0 for each dimension
        ):
        super().__init__()
        self.observation_space = observation_max
        self.action_space = action_max

        try:
            self.observation_size = np.sum(list(observation_max))
        except:
            self.observation_size = observation_max

        try:
            self.action_size = np.sum(list(action_max))
        except:
            self.action_size = action_max
        
    def create_model(
        self,
        hidden_layer_list = list,
        activation_list = list,
        ):
        if len(hidden_layer_list) != len(activation_list):
            raise Exception(
                'Hidden layer list and activation list must be the same size.'
                )

        self.create_neuron_stack(
            hidden_layer_list,
            activation_list,
        )
        self.model_policy = DRL_NN.DRL_NN(self.neuron_stack)
        self.model_target = DRL_NN.DRL_NN(self.neuron_stack)

        self.model_target.load_state_dict(
            self.model_policy.state_dict()
        )

    def create_neuron_stack(
        self,
        hidden_layer_list = list,
        activation_list = list,
        ):
        self.neuron_stack = nn.Sequential(
            nn.Linear(
                self.observation_size,
                hidden_layer_list[0]
            ),
            activation_list[0],
        )

        if len(hidden_layer_list) > 1:
            for layer in range(1,len(hidden_layer_list)):
                self.neuron_stack.append(
                    nn.Linear(
                        hidden_layer_list[layer-1],
                        hidden_layer_list[layer]
                    )
                )
                self.neuron_stack.append(
                    activation_list[layer]
                )
        
        self.neuron_stack.append(
            nn.Linear(
                hidden_layer_list[-1],
                self.action_size
            )
        )

    def observation_space_tensor(
            self,
            observation
        ):
        if type(observation) == tuple:
            OHE_input_tens = torch.tensor([])

            for position, observation_ in enumerate(observation):
                OHE_input_tens_ = torch.zeros([self.observation_space[position]])
                OHE_input_tens_[observation_] = 1
                OHE_input_tens = torch.cat(
                    (
                        OHE_input_tens,
                        OHE_input_tens_
                    ),
                    0
                )
        else:
            OHE_input_tens = torch.zeros(self.observation_space)
            OHE_input_tens[observation] = 1

        return OHE_input_tens

    def game_cycle(
        self,
        games_per_epoch,
        gamma,
        env,
        ):
        memory = []
        for game in range(games_per_epoch):
            observation, info = env.reset()
            done = 0
            while not done:
                input_tens = self.observation_space_tensor(observation)
                gamma_against = np.random.random()
                if gamma_against>gamma:
                    output_tens = self.model_policy.forward_only(input_tens)
                    action = output_tens.detach().numpy().argmax()
                else:
                    action = np.random.randint(0,2)

                new_observation, reward, terminated, truncated, info = env.step(action)

                with torch.no_grad():
                    memory_input = np.array([
                        input_tens,
                        action,
                        self.observation_space_tensor(new_observation),
                        reward,
                        terminated or truncated,
                    ],
                    dtype=object
                    )

                memory.append(memory_input)

                observation = new_observation

                done = terminated or truncated
        return memory
    
    def learning_cycle(
        self,
        memory,
        discount_factor,
        criterion,
        optimizer,
        ):
        policy_q_list = []
        target_q_list = []

        for input_space, action, output_space, reward, done in memory:
            if done == True:
                quality = reward
            else:
                quality = reward + max(
                                        self.model_policy.forward_only(
                                                output_space
                                        )
                                    )*discount_factor

            policy_q = self.model_policy.forward_only(input_space)
            target_q = self.model_target.forward_only(input_space)

            target_q[action] = quality

            policy_q_list.append(policy_q)
            target_q_list.append(target_q)

        self.model_policy.learn(
            y_hat=torch.stack(policy_q_list),
            y=torch.stack(target_q_list),
            criterion=criterion,
            optimizer=optimizer,
        )

        self.model_target.load_state_dict(self.model_policy.state_dict())

    def train(
        self,
        env,
        epochs,
        games_per_epoch,
        criterion,
        optimizer,
        gamma,
        discount_factor,
        validation = bool,
        validation_env = None,
        validation_size = None,
        ):
        self.gamma_store = []
        if validation:
            self.validation_dict = {}
        for epoch in range(epochs):
            gamma_ = gamma*(
                (
                    (
                        np.log2(
                            -1*(
                                (
                                    epoch+1
                                )/epochs
                            )+1
                        )
                    )/5
                ) + 1
            )

            memory = self.game_cycle(
                games_per_epoch=games_per_epoch,
                gamma=gamma_ if gamma_ > 0 else 0,
                env=env,
            )

            self.learning_cycle(
                memory=memory,
                discount_factor=discount_factor,
                criterion=criterion,
                optimizer=optimizer,
            )

            if validation:
                self.validation_cycle(
                    validation_env,
                    epoch,
                    validation_size,
                )
    
    def validation_cycle(
            self,
            env,
            epoch = int,
            validation_size = int,
        ):
        self.model_target.eval()
        wins = 0
        losses = 0
        for validation_games in range(validation_size):
            done = 0
            observation, info = env.reset()
            while not done:
                input_tens = self.observation_space_tensor(observation)

                output_tens = self.model_target.forward_only(input_tens)
                action = output_tens.detach().numpy().argmax()

                new_observation, reward, terminated, truncated, info = env.step(action)

                observation = new_observation

                done = terminated or truncated

            if reward > 0.5:
                wins += reward
            if reward < -0.5:
                losses -= reward
        print(f'Validation: (Epoch: {epoch+1}; Wins: {wins}; Losses: {losses})')
        self.validation_dict[epoch+1] = {
            'Wins': wins,
            'Losses': losses
        }
        
    def evaluate(
            self,
            observation,
        ):
        decision = np.argmax(
            self.model_target.forward_only(
                self.observation_space_tensor(
                    observation
                )
            ).detach().numpy()
        )
        return decision

# %%
