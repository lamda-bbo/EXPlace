import torch as th
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import time
from torchvision.models import resnet18
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pdb

from model.cnn import MyCNN, MyCNNCoarse
from model.actor import Actor
from model.critic import Critic


class PPOAgent():
    def __init__(self, args):
        self.args = args
        self.resnet = resnet18(pretrained=True)
        actor_arch = getattr(args, 'actor_arch', 'parallel')
        if actor_arch == 'sequential':
            k = getattr(args, 'local_summary_k', 4)
            canvas_num = 2 if getattr(args, 'prototype_flag', False) else 1
            self.cnn = MyCNN(args=args, out_channels=k).to(args.device)
            self.cnn_coarse = MyCNNCoarse(args=args, res_net=self.resnet, input_dim=canvas_num + k).to(args.device)
        else:
            self.cnn = MyCNN(args=args).to(args.device)
            self.cnn_coarse = MyCNNCoarse(args=args, res_net=self.resnet).to(args.device)

        self.actor = Actor(args=args, cnn=self.cnn, cnn_coarse=self.cnn_coarse).float().to(args.device)
        self.critic = Critic(args=args, cnn_coarse=self.cnn_coarse).float().to(args.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), args.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), args.lr)

        self.buffer = []
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size * args.n_macro
        self.counter = 0
        self.training_step = 0

        self.training = False
    
    def select_action(self, state:np.array):
        """
        Select actions for a batch of environments.

        Args:
            state (np.array): Batched state input, shape [batch_size, ...].

        Returns:
            actions (list): List of selected actions for each environment, length=batch_size.
            action_log_probs (list): List of log probabilities for each selected action, length=batch_size.

        This method converts the input numpy array to a torch tensor, computes action probabilities
        using the actor network, samples or selects the most probable action for each environment
        (depending on training/eval mode), and returns the results as Python lists.
        """
        # state: [batch, ...], do not add unsqueeze(0)
        state = th.from_numpy(state).float().to(self.args.device)
        with th.no_grad():
            action_prob = self.actor(state)  # [batch, N]
        dist = Categorical(action_prob)
        if self.training:
            action = dist.sample()  # [batch]
        else:
            action = dist.probs.argmax(dim=1)  # [batch]
        action_log_prob = dist.log_prob(action)  # [batch]
        return action.cpu().numpy().tolist(), action_log_prob.cpu().numpy().tolist()
    
    def update(self):
        if self.counter % self.buffer_capacity == 0:
            t_start = time.time() #########
            state = th.tensor(np.concatenate([t.state for t in self.buffer], axis=0), dtype=th.float32)
            action = th.tensor(np.array([t.action for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            reward = th.stack([t.reward.to(self.args.device) for t in self.buffer]).view(-1, 1)
            # reward = th.tensor(np.array([t.reward for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            old_action_log_prob = th.tensor(np.array([t.action_log_prob for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            done = th.tensor(np.array([t.done for t in self.buffer], dtype=np.int32), dtype=th.int32).view(-1, 1).to(self.args.device)
            del self.buffer[:]
            
            # Calculate cumulative returns
            target_list = []
            target = 0
            for i in range(reward.shape[0]-1, -1, -1):
                if done[i, 0] == 1:
                    target = 0
                r = reward[i, 0].item()
                target = r + self.args.gamma * target
                target_list.append(target)
            target_list.reverse()
            target_v_all = th.tensor(np.array([t for t in target_list]), dtype=th.float32).view(-1, 1).to(self.args.device)
            
            actor_loss_lst  = []
            critic_loss_lst = []
            for _ in range(self.args.epoch):
                for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                    disable = self.args.disable_tqdm):
                    self.training_step += 1
                    action_prob = self.actor(state[index].to(self.args.device))
                    dist = Categorical(action_prob)
                    action_log_prob = dist.log_prob(action[index].squeeze())
                    ratio = th.exp(action_log_prob - old_action_log_prob[index].squeeze())
                    target_v = target_v_all[index]
                    critic_output = self.critic(state[index].to(self.args.device))
                    advantage = (target_v - critic_output).detach()
                    L1 = ratio * advantage.squeeze() 
                    L2 = th.clamp(ratio, 1-self.args.clip_param, 1+self.args.clip_param) * advantage.squeeze() 
                    action_loss = -th.min(L1, L2).mean() 

                    self.actor_opt.zero_grad()
                    action_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                    self.actor_opt.step()

                    value_loss = nn.functional.smooth_l1_loss(self.critic(state[index].to(self.args.device)), target_v)
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                    self.critic_opt.step()

                    actor_loss_lst.append(action_loss.cpu().item())
                    critic_loss_lst.append(value_loss.cpu().item())

            t_end = time.time() #########
            print("Training time: {:.2f}s".format(t_end - t_start)) #########
            return t_end - t_start, np.mean(actor_loss_lst), np.mean(critic_loss_lst)   
        else:
            return 0, -1, -1

    def store_transition(self, transition):
        self.counter += 1
        self.buffer.append(transition)

    def save_model(self, episode, best_so_far=False, path=None, filename=None):
        base_path = path if path is not None else self.args.log_dir
        if filename is not None:
            file_name = filename
        else:
            if best_so_far:
                file_name = f'best_model_ep{episode}.pt'
            else:
                file_name = f'model_ep{episode}.pt'
        os.makedirs(base_path, exist_ok=True)
        target_path = os.path.join(base_path, file_name)
        th.save(
            {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'episode': episode
            },
            target_path
        )
    
    def load_model(self, checkpoint_path):
        checkpoint = th.load(checkpoint_path, map_location=th.device(self.args.device))
        if 'actor' in checkpoint and 'critic' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
        else:
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()