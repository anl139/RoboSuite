import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import os

class CriticNetwork(nn.Module):
    def __init__(self, input_dims,n_actions,fc1_dims=256,fc2_dims=128, name='critic',checkpoint_dir='tmp/td3', learning_rate=1e-3):
        super(CriticNetwork,self).__init__()

        if isinstance(input_dims, (tuple, list)):
            self.input_dims = int(input_dims[0])
        else:
            self.input_dims = int(input_dims)

        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims,1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay = 0.005)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Created Critic Network on Device {self.device}")
        self.to(self.device)
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
    
        action_value = self.fc1(T.cat([state,action], dim = 1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)
        return q1
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims,n_actions = 2,fc1_dims=256,fc2_dims=128,  name='actor',checkpoint_dir='tmp/td3', learning_rate=10e-3):
        super(ActorNetwork,self).__init__()
        if isinstance(input_dims, (tuple, list)):
            self.input_dims = int(input_dims[0])
        else:
            self.input_dims = int(input_dims)

        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims,self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,weight_decay = 0.005)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Created Actor Network on Device {self.device}")
        self.to(self.device)

    def forward(self, state):
        # accepts (batch, dim) or (dim,) as tensor
        single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single = True

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.output(x))

        return x.squeeze(0) if single else x  # single -> (n_actions,), batch -> (batch, n_actions)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
