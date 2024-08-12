# TODO
# A. frame stack 5 -> 10
# B. depth quantization
# C. lower resolution (28x28)

# %%
from collections import deque
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm.notebook import trange
import wandb

# from simpl.collector import ConcurrentCollector, TimeLimitCollector, GPUWorker, Buffer
# from simpl.nn import itemize
# from simpl.math import discount
# from simpl.rl.policy import ContextTruncatedNormalMLPPolicy
# from simpl.rl.qf import MLPQF

# %%
rollout_dir = '../collect/v3'
rollouts = [
    torch.load(f'{rollout_dir}/{filename}')
    for filename in os.listdir(rollout_dir) if filename[-3:] == '.pt'
]
lengths = [
    len(rollout['actions'])
    for rollout in rollouts
]

# %%
data_states = torch.as_tensor(np.concatenate([
    rollout['states'][1:]
    for rollout in rollouts
]), dtype=torch.float32).squeeze(-1)

data_actions = torch.as_tensor(np.concatenate([
    np.array(rollout['actions'])
    for rollout in rollouts
]), dtype=torch.long)

data_states.shape, data_actions.shape
# %%
skill_length = 5
frame_stack = 5

# stacked_data_states = torch.stack([torch.cat((
#     torch.zeros((i, 84, 84)),
#     data_states,
#     torch.zeros((5-i, 84, 84))
#     )) for i in range(frame_stack)]).swapaxes(0, 1).contiguous()

data_available_indices = []
rollout_start_idx = 0
for length in lengths:
    available_indices = torch.arange(frame_stack-1, length-skill_length+1)
    data_available_indices.append(rollout_start_idx + available_indices)
    rollout_start_idx += length
data_available_indices = torch.cat(data_available_indices)

# %%
def add_random_boxes(img, max_k, size=16):
    h,w = size,size
    img_size = img.shape[-2]
    boxes = []
    n_k = np.random.randint(max_k)
    for k in range(n_k):
        y,x = np.random.randint(0,img_size-w,(2,))
        
        img[:, y:y+h,x:x+w] = 0
        boxes.append((x,y,h,w))
    return img


def quantize(img, num_level):
    alpha_q = 0
    beta_q = num_level - 1
    alpha = 0
    beta = 255

    s = (beta - alpha) / (beta_q - alpha_q)
    z = round((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    return (img / s).round() + z

def quantize_with_thresholds(img, thresh_list):
    new_img = torch.zeros_like(img)
    n_levels = len(thresh_list) + 1
    for i in range(1, n_levels):
        new_img[img > thresh_list[i-1]] = i
    return new_img

# %%
from torch.utils.data import Dataset, BatchSampler, RandomSampler, DataLoader
from torchvision.transforms import Compose, RandomRotation, CenterCrop, Resize

class TrajDataset(Dataset):
    augment = Compose([
        RandomRotation(5),
        CenterCrop(75),
        # Resize(28)
        Resize(84)
    ])

    val_augment = Compose([
        CenterCrop(75),
        # Resize(28)
        Resize(84)
    ])
    
    def __init__(self, indices, validate=False):
        self.indices = indices
        self.validate = validate

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx_of_available_indices):
        if type(idx_of_available_indices) != int:
            raise
        idx = self.indices[idx_of_available_indices]
        seq_state = data_states[idx-frame_stack+1:idx+skill_length]
        
        stack_seq_state = []
        for stack_i in range(frame_stack):
            stack_seq_state.append(seq_state[stack_i:stack_i+skill_length])
        stack_seq_state = torch.stack(stack_seq_state).swapaxes(0, 1)
        
        seq_action = data_actions[idx:idx+skill_length]
        
        if not self.validate: # augmentation
            stack_seq_state = self.augment(add_random_boxes(stack_seq_state, 3))
        else:
            stack_seq_state = self.augment(stack_seq_state)

        # stack_seq_state = quantize(stack_seq_state, 10)
        # return stack_seq_state / 9, seq_action

        thresholds = [10 * i for i in range(10,20)]
        stack_seq_state = quantize_with_thresholds(stack_seq_state, thresholds)
        return stack_seq_state / len(thresholds), seq_action

        return stack_seq_state / 255, seq_action

    # def __getitem__(self, idx_of_available_indices):
    #     if type(idx_of_available_indices) != int:
    #         raise
    #     idx = data_available_indices[idx_of_available_indices]
    #     stack_seq_state = stacked_data_states[idx-1:idx+skill_length+frame_stack-1]
        
    #     seq_action = data_actions[idx:idx+skill_length]
        
    #     stack_seq_state = self.augment(add_random_boxes(stack_seq_state, 3))
        
    #     return stack_seq_state / 255, seq_action

# np_random = np.random.RandomState(seed=4)
# np.random.permutation
# data_available_indices = data_available_indices[torch.randperm(data_available_indices.nelement())]
training_indices, val_indices = data_available_indices[:3000], data_available_indices[3000:]

# training_indices = data_available_indices[:1500]
dataset, val_dataset = TrajDataset(training_indices), TrajDataset(val_indices, True)
print(len(dataset), len(val_dataset))
# dataset, val_dataset = torch.utils.data.random_split(dataset, [3000, len(dataset)-3000])

# %%
import torch.nn as nn
from simpl.nn import MLP

from simpl.nn import ToDeviceMixin
import torch.distributions as torch_dist

from simpl.math import inverse_softplus, inverse_sigmoid


class SkillEncoder(ToDeviceMixin, nn.Module):
    def __init__(self, action_dim, z_dim, hidden_dim, n_lstm, n_mlp_hidden):
        super().__init__()
        
        self.action_dim = action_dim
        
        self.lstm = nn.LSTM(
            action_dim,
            hidden_dim, n_lstm, batch_first=True
        )
        self.mlp = MLP([hidden_dim]*n_mlp_hidden + [2*z_dim], 'relu')
        
        self.register_buffer('prior_loc', torch.zeros(z_dim))
        self.register_buffer('prior_scale', torch.ones(z_dim))
        # self.register_buffer('h0', torch.zeros(n_lstm, hidden_dim))
        # self.register_buffer('c0', torch.zeros(n_lstm, hidden_dim))

    @property
    def prior_dist(self):
        return torch_dist.Independent(torch_dist.Normal(self.prior_loc, self.prior_scale), 1)
    
        
    def dist(self, batch_seq_action):
        # batch_h0 = self.h0[:, None, :].expand(-1, len(batch_seq_state), -1)
        # batch_c0 = self.c0[:, None, :].expand(-1, len(batch_seq_state), -1)
        
        batch_seq_onehot_action = F.one_hot(batch_seq_action, num_classes=self.action_dim).float()
        batch_seq_out, _ = self.lstm(batch_seq_onehot_action)
        batch_last_out = batch_seq_out[:, -1, :]
        batch_loc, batch_pre_scale = self.mlp(batch_last_out).chunk(2, dim=-1)
        batch_scale = F.softplus(batch_pre_scale)
        
        return torch_dist.Independent(
            torch_dist.Normal(batch_loc, batch_scale)
        , 1)

class PriorPolicy(ToDeviceMixin, nn.Module):
    def __init__(self, state_shape, z_dim, hidden_dim, n_hidden,
                 min_scale=0.001, max_scale=None, init_scale=0.1):
        super().__init__()
        
        assert state_shape == (5, 84, 84)
        assert hidden_dim == 128
        
        self.z_dim = z_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
        )
        self.mlp = MLP([128]*n_hidden + [2*z_dim], 'relu')
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        if max_scale is None:
            self.pre_init_scale = inverse_softplus(init_scale)
        else:
            self.pre_init_scale = inverse_sigmoid(init_scale / max_scale)
        
    def dist(self, batch_state):
        input_dim = batch_state.dim()
        if input_dim > 4:
            batch_shape = batch_state.shape[:-3] 
            data_shape = batch_state.shape[-3:]
            batch_state = batch_state.view(-1, *data_shape)
        batch_h = self.conv_net(batch_state)[..., 0, 0]
        batch_loc, batch_pre_scale = self.mlp(batch_h).chunk(2, dim=-1)

        if self.max_scale is None:
            batch_scale = self.min_scale + F.softplus(self.pre_init_scale + batch_pre_scale)
        else:
            batch_scale = self.min_scale + self.max_scale*torch.sigmoid(self.pre_init_scale + batch_pre_scale)
        
        if input_dim > 4:
            batch_loc = batch_loc.view(*batch_shape, self.z_dim)
            batch_scale = batch_scale.view(*batch_shape, self.z_dim)
        
        return torch_dist.Independent(
            torch_dist.Normal(batch_loc, batch_scale)
        , 1)
        
# class LowPolicy(ToDeviceMixin, nn.Module):
#     def __init__(self, state_shape, action_dim, z_dim, hidden_dim, n_hidden):
#         super().__init__()
        
#         assert state_shape == (5, 84, 84)
#         assert hidden_dim == 128
        
#         self.action_dim = action_dim
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(5, 32, kernel_size=4, stride=3),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=3),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1),
#         )
#         self.mlp = MLP([128 + z_dim] + [hidden_dim]*(n_hidden-1) + [action_dim], 'relu')
        
#     def dist(self, batch_state, batch_z):
#         input_dim = batch_state.dim()
#         if input_dim > 4:
#             batch_shape = batch_state.shape[:-3]
            
#             data_shape = batch_state.shape[len(batch_shape):]
#             batch_state = batch_state.view(-1, *data_shape)

#             data_shape = batch_z.shape[len(batch_shape):]
#             batch_z = batch_z.reshape(-1, *data_shape)
        
#         batch_h = self.conv_net(batch_state)[..., 0, 0]
#         batch_h_z = torch.cat([batch_h, batch_z], dim=-1)
#         batch_logits = self.mlp(batch_h_z)
        
#         if input_dim > 4:
#             batch_logits = batch_logits.view(*batch_shape, self.action_dim)
        
#         return torch_dist.Categorical(logits=batch_logits)

# class LowPolicy(ToDeviceMixin, nn.Module):
#     def __init__(self, action_dim, z_dim, hidden_dim, n_lstm):
#         super().__init__()
        
#         assert hidden_dim == 128
         
#         self.action_dim = action_dim
#         self.lstm = nn.LSTM(
#             z_dim,
#             hidden_dim, n_lstm, batch_first=True, proj_size=action_dim,
#         )
        
#     def dist(self, batch_z):
#         batch_logits, _  = self.lstm(batch_z)
#         batch_logits = F.log_softmax(batch_logits, dim=-1)
#         return torch_dist.Categorical(logits=batch_logits)

from vit_pytorch import ViT
# modified output ViT; from (batch, num_class) to (batch, dim)

class LowPolicy(ToDeviceMixin, nn.Module):
    def __init__(self, action_dim, z_dim, hidden_dim, n_lstm):
        super().__init__()
        
        assert hidden_dim == 128
        
        self.action_dim = action_dim
        self.vit = ViT(
            image_size = 84,
            patch_size = 21,
            num_classes = 3,
            dim = 256,
            depth = 3,
            heads = 8,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            channels = 5,
        )

        self.mlp = MLP([256 + z_dim] + [256]*(2-1) + [action_dim], 'relu')
        
    def dist(self, batch_state, batch_z):
        # batch_state = torch.flatten(batch_state, start_dim=0, end_dim=1)
        # batch_z = torch.flatten(batch_z, start_dim=0, end_dim=1)
        data_shape = batch_state.shape[:2]
        batch_state = batch_state.reshape(-1, *batch_state.shape[2:])
        batch_z = batch_z.reshape(-1, *batch_z.shape[2:])
        batch_logits = self.vit(batch_state)
        batch_logits = torch.cat([batch_logits, batch_z], dim=1)
        batch_logits = self.mlp(batch_logits)
        batch_logits = F.log_softmax(batch_logits, dim=-1)
        batch_logits = batch_logits.reshape(*data_shape, -1)
        return torch_dist.Categorical(logits=batch_logits)


# %%
config = dict(
    encoder=dict(hidden_dim=256, n_lstm=2, n_mlp_hidden=2),
    prior_policy=dict(hidden_dim=128, n_hidden=2, init_scale=1, max_scale=2),
    low_policy=dict(hidden_dim=128, n_lstm=2,),
    batch_size=128,
    z_dim=10,
    reg_scale=0.1,
)
gpu = 7
state_shape = (5, 84, 84)
action_dim = 3
z_dim = config['z_dim']

filename = 'pretrain.spirl_v5_skill5_z10'

# %%
encoder = SkillEncoder(action_dim, z_dim, **config['encoder']).to(gpu)
prior_policy = PriorPolicy(state_shape, z_dim, **config['prior_policy']).to(gpu)
low_policy = LowPolicy(action_dim, z_dim, **config['low_policy']).to(gpu)

encoder_optim = torch.optim.Adam(encoder.parameters(), lr=3e-4)
prior_policy_optim = torch.optim.Adam(prior_policy.parameters(), lr=3e-4)
low_policy_optim = torch.optim.Adam(low_policy.parameters(), lr=3e-4)

skill_prior_dist = torch_dist.Independent(torch_dist.Normal(
    torch.zeros(z_dim).to(gpu),
    torch.ones(z_dim).to(gpu),
), 1)

# %%
loader = DataLoader(
    dataset, 
    batch_size=config['batch_size'],
    drop_last=True,
    shuffle=True,
   num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    # batch_size=config['batch_size'],
    batch_size=len(val_dataset),
    drop_last=True,
    shuffle=False,
    num_workers=4
)

# %%
wandb.init(project='drone', entity='mlai-rl', config=config)

####
wandb.run.name = f'{filename}_{wandb.run.id}_B100200'
####
wandb.run.save()
save_filename = wandb.run.name.split('/')[-1] + '.pt'

# %%
def validation(encoder, prior_policy, low_policy, log):

    num_success = 0
    val_recon_loss = 0
    val_prior_loss = 0
    val_reg_loss = 0
    val_loss = 0
    val_action_accuracy = 0

    with torch.no_grad():
        for step_i, (batch_seq_state, batch_seq_action) in enumerate(val_loader):
            batch_seq_state = batch_seq_state.to(gpu)
            batch_seq_action = batch_seq_action.to(gpu)
            
            batch_skill_dist = encoder.dist(batch_seq_action)
            batch_skill_prior_dist = prior_policy.dist(batch_seq_state[:, 0, :])
            
            batch_skill = batch_skill_dist.rsample()
            batch_seq_skill = batch_skill[:, None, :].expand(-1, skill_length, -1)

            batch_seq_policy_dist = low_policy.dist(batch_seq_state, batch_seq_skill)
            # batch_seq_policy_dist = low_policy.dist(batch_seq_skill)

            recon_loss = -batch_seq_policy_dist.log_prob(batch_seq_action).mean((0, 1))
            
            reg_loss = torch_dist.kl_divergence(
                batch_skill_dist,
                skill_prior_dist
            ).mean(0)
            
            prior_loss = - batch_skill_prior_dist.log_prob(batch_skill_dist.sample()).mean(0)
            # batch_skill_dist.base_dist.loc = batch_skill_dist.base_dist.loc.detach()
            # batch_skill_dist.base_dist.scale = batch_skill_dist.base_dist.scale.detach()
            # prior_loss = torch_dist.kl_divergence(
            #     batch_skill_dist,
            #     batch_skill_prior_dist
            # ).mean(0)
            
            loss = recon_loss + prior_loss + config['reg_scale']*reg_loss

            val_recon_loss += recon_loss
            val_reg_loss += reg_loss
            val_prior_loss += prior_loss
            val_loss += loss
            
            action_accuracy = (batch_seq_policy_dist.logits.argmax(-1) == batch_seq_action).float().mean()
            val_action_accuracy += action_accuracy

        log.update({
            'val_action_acc': val_action_accuracy / (step_i+1),
            'val_recon_loss': val_recon_loss / (step_i+1),
            'val_reg_loss': val_reg_loss / (step_i+1),
            'val_prior_loss': val_prior_loss / (step_i+1),
            'val_loss': val_loss / (step_i+1),
            'val_skill_prior_ent': -batch_skill_prior_dist.log_prob(batch_skill_dist.mean).mean(0),
            })

        return log

# %%
import torch.nn.functional as F

for epoch_i in range(1, 5001):
    log = {'epoch_i': epoch_i}

    for step_i, (batch_seq_state, batch_seq_action) in enumerate(loader):
        batch_seq_state = batch_seq_state.to(gpu)
        batch_seq_action = batch_seq_action.to(gpu)
        
        batch_skill_dist = encoder.dist(batch_seq_action)
        batch_skill_prior_dist = prior_policy.dist(batch_seq_state[:, 0, :])
        
        batch_skill = batch_skill_dist.rsample()
        batch_seq_skill = batch_skill[:, None, :].expand(-1, skill_length, -1)

        batch_seq_policy_dist = low_policy.dist(batch_seq_state, batch_seq_skill)
        # batch_seq_policy_dist = low_policy.dist(batch_seq_skill)

        recon_loss = -batch_seq_policy_dist.log_prob(batch_seq_action).mean((0, 1))
        
        reg_loss = torch_dist.kl_divergence(
            batch_skill_dist,
            skill_prior_dist
        ).mean(0)
        
        
        if epoch_i < 20:
            prior_loss = 0
        else:
            prior_loss = - batch_skill_prior_dist.log_prob(batch_skill_dist.sample()).mean(0)
            # batch_skill_dist.base_dist.loc = batch_skill_dist.base_dist.loc.detach()
            # batch_skill_dist.base_dist.scale = batch_skill_dist.base_dist.scale.detach()
            # prior_loss = torch_dist.kl_divergence(
            #     batch_skill_dist,
            #     batch_skill_prior_dist
            # ).mean(0)
        loss = recon_loss + prior_loss + config['reg_scale']*reg_loss
        
        encoder_optim.zero_grad()
        low_policy_optim.zero_grad()
        prior_policy_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        low_policy_optim.step()
        prior_policy_optim.step()

    # validation
    log = validation(encoder, prior_policy, low_policy, log)    
    
    action_accuracy = (batch_seq_policy_dist.logits.argmax(-1) == batch_seq_action).float().mean()
    
    log.update({
        'loss': loss,
        'recon_loss': recon_loss,
        'prior_loss': prior_loss,
        'reg_loss': reg_loss,
        'mean_encoder_scale': batch_skill_dist.base_dist.scale.mean(),
        'mean_prior_policy_scale': batch_skill_prior_dist.base_dist.scale.mean(),
        'action_accuracy': action_accuracy,
        'skill_prior_ent': -batch_skill_prior_dist.log_prob(batch_skill_dist.mean).mean(0)
    })

    wandb.log(log)

    # if epoch_i % 250 == 0:
    #     torch.save({
    #         'prior_policy': prior_policy,
    #         'low_policy': low_policy,
    #         'encoder': encoder,
    #     }, f'{wandb.run.name}_{epoch_i}.pt')


# %%
torch.save({
    'prior_policy': prior_policy,
    'low_policy': low_policy,
    'encoder': encoder,
}, f'{wandb.run.name}.pt')