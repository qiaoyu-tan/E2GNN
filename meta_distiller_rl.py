import numpy as np
from torch.nn import BatchNorm1d, Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN, SAGE, GAT, APPNPM, SGC


class BaseSampler:
    def __init__(self, device, args):
        raise NotImplementedError

    def learn(self, model, data, y_soft_labels, y_soft, train_mask, device, warmup=None):
        pass


class OriginalSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device

    def sample(self, x):
        num_node, _ = x.shape
        reliable_soft = torch.ones(num_node, dtype=torch.float).view(-1, 1).to(self.device)
        return reliable_soft


class RandomSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device

    def sample(self, data, training=False):

        # Random
        num_node, _ = data.x.shape
        reliable_soft = torch.randint(2, (num_node, 1)).to(torch.float).to(self.device)
        # num_node.. 0,1

        return reliable_soft


class SinglePolicyGradientSampler(BaseSampler):
    def __init__(self, device, out_dim,args):
        # Hyperparameters
        self.learning_rate = 0.005
        # self.start_entropy = 0.05 # Larger -> uniform
        self.start_entropy = 0.2 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 5 # Iterations
        self.gama = args.gama
        self.clip_norm = 30
        self.epochs = 20
        self.out_dim=out_dim

        self.entropy_coefficient = self.start_entropy

        self.device = device
        # self.net = SinglePolicyNet(
        #     args.hidden_mlp,
        #     2
        # ).to(device)
        self.net = SinglePolicyNetGNN(
            input_dim=args.hidden_mlp,
            hidden_dim=args.hidden_mlp,
            num_layers=args.meta_layer,
            out_dim=out_dim
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

    def sample(self, x, train_mask=None, training=True, return_logits=False):
        # num_node, dim = x.shape
        logits = self.net.forward(x)
        if train_mask is not None:
            logits['policy'] = logits['policy'][train_mask]
        # logits = logits[train_mask]
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        if return_logits:
            return action, logits
        else:
            return action

    def compute_reward(self, y_soft_, pred_stu, action, y_truth):
        y_soft_act = torch.stack([y_soft_[j,i,:] for i, j in enumerate(action)], dim=0)
        loss2 = F.kl_div(pred_stu.log_softmax(dim=-1), y_soft_act, reduction='none', log_target=False).sum(
            -1)  # (u_mask,)
        loss1 = F.cross_entropy(pred_stu, y_truth, reduction='none')
        reward = loss2 + self.gama * loss1
        return reward

    def learn(self, model, data, y_soft_labels, y_soft, train_mask, device, warmup=None):
        model.eval()
        y_soft_train = y_soft[train_mask]
        y_truth = data.y[train_mask]
        data_stu = data.clone()
        y_soft_ = y_soft_labels[:, train_mask, :]
        pred_stu = model(data.x)[train_mask].detach()
        y_soft_ = torch.cat([y_soft_, pred_stu.view(1, y_truth.shape[0], -1)], dim=0)
        with torch.no_grad():
            # h = model.encode(data.x[train_mask])
            h = model.encode(data.x)
            # h = torch.cat([h, y_soft_train], dim=1)
        data_stu.x = h
        count = 0
        stats = {
            "policy_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }
        for epoch in range(self.epochs):
            action, logits = self.sample(data_stu, train_mask, return_logits=True)
            # action, logits = self.sample(h, return_logits=True)
            reward,reward_ratio = compute_reward_gmt(action, y_soft_train, device)
            reward *= self.compute_reward(y_soft_, pred_stu, action, y_truth)

            policy_logits = logits["policy"]

            # Normalize rewards
            reward=reward.float()
            reward = (reward - reward.mean()) / (reward.std() + 10 - 8)
            # Policy gradient
            policy_loss = compute_policy_loss(policy_logits, action, reward)

            # Entropy
            entropy_loss = compute_entropy_loss(policy_logits)

            if warmup is not None:
                total_loss = entropy_loss
            else:
                total_loss = policy_loss + entropy_loss * self.entropy_coefficient

            self.optimizer.zero_grad()
            total_loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            # print('### Epoch: {} reward_ratio: {} action: {} policy_loss: {} entropy_loss: {}'.format(epoch, reward_ratio, action.sum()//action.shape[0], policy_loss.item(), entropy_loss.item()))

            stats["policy_loss"].append(policy_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

            count += 1
            if warmup is None and count >= self.num_updates:
                break
            elif warmup is not None and count >= warmup:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        # print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)


class SingleActorCriticeSampler(BaseSampler):
    def __init__(self, device, args):
        # Hyperparameters
        self.learning_rate = 0.01
        self.start_entropy = 0.05 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 10 # Iterations
        self.clip_norm = 30
        self.value_coeficient = 0.5

        self.entropy_coefficient = self.start_entropy

        self.device = device
        # self.net = SinglePolicyValueNet(
        #     args.hidden_channels,
        #     args.num_layers
        # ).to(device)
        self.net = SinglePolicyNetGNN(
            args.hidden_channels,
            args.num_layers
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

    def sample(self, h_edge_0, h_edge_1, training=True, return_logits=False):
        num_edges, max_hop, _ = h_edge_0.shape
        edge_features = torch.cat((h_edge_0[:,0].detach(), h_edge_1[:,0].detach()), dim=1)
        logits = self.net.forward(edge_features)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        if return_logits:
            return action, logits
        else:
            return action

    def learn(self, model, data, y_soft_labels, pos_valid_edge, valid_index_loader, warmup=None):
        model.eval()
        data_stu = data.clone
        with torch.no_grad():
            h = model(data.x, data.edge_index)
        data_stu.x = h
        count = 0
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }
        for perm in valid_index_loader:
            valid_edge = pos_valid_edge[perm].t()
            valid_edge_neg = generate_neg_sample(pos_valid_edge, data.num_nodes, data.x.device, perm.shape[0])

            pos_num, neg_num = valid_edge.shape[1], valid_edge_neg.shape[1]
            out, action, logits = model.compute_pred_and_logits(
                h,
                torch.cat((valid_edge, valid_edge_neg), dim=1),
                self,
            )
            policy_logits = logits["policy"]
            value_logits = logits["value"]
            out = out.squeeze()
            pos_out, neg_out = out[:pos_num], out[-neg_num:]
            pos_reward = torch.log(pos_out + 1e-15)
            neg_reward = torch.log(1 - neg_out + 1e-15)
            reward = torch.cat((pos_reward, neg_reward))
            advantage = reward - value_logits.detach().squeeze()

            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 10-8)

            # Policy gradient
            policy_loss = compute_policy_loss(policy_logits, action, advantage)

            # Value loss
            value_loss = compute_value_loss(reward - value_logits) * self.value_coeficient

            # Entropy
            entropy_loss = compute_entropy_loss(policy_logits)

            if warmup is not None:
                total_loss = entropy_loss
            else:
                total_loss = policy_loss + value_loss + entropy_loss * self.entropy_coefficient

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

            count += 1
            if warmup is None and count >= self.num_updates:
                break
            elif warmup is not None and count >= warmup:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)


samplers = {
    "original": OriginalSampler,
    "random": RandomSampler,
    "single_policy_gradient": SinglePolicyGradientSampler,
    "single_actor_critic": SingleActorCriticeSampler
}


def get_sampler(name):
    if name not in samplers:
        return ValueError("Sampler not supported. Choices: {}".format(samplers.keys()))
    return samplers[name]


class SinglePolicyValueNet(nn.Module):
    def __init__(self, dim, num_layers):
        super(SinglePolicyValueNet, self).__init__()

        self.fc1 = nn.Linear(dim*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.policy_head = nn.Linear(32, num_layers)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        return {"policy": policy_logits, "value": value_logits}


class SinglePolicyNet(nn.Module):
    def __init__(self, dim, num_layers, dropout=0.5):
        super(SinglePolicyNet, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(dim, 64)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_layers)
        self.norms = torch.nn.ModuleList()
        batch_norm_kwargs = {}

        for hidden_channels in [64, 32]:
            norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            self.norms.append(norm)

    def forward(self, x):
        x = F.relu(self.norms[0](self.fc1(x)))

        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.fc2(x))
        x = F.relu(self.norms[1](self.fc3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return {"policy": x}


class SinglePolicyNetGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,out_dim,num_layers=1, dropout=0.5):
        super(SinglePolicyNetGNN, self).__init__()
        self.dropout = dropout
        #self.out_dim=out_dim
        self.conv = GCN(input_dim, hidden_dim, out_dim, num_layers, norm=1)
        # self.fc1 = nn.Linear(dim, 64)
        # # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 32)
        # self.fc4 = nn.Linear(32, num_layers)
        # self.norms = torch.nn.ModuleList()
        # batch_norm_kwargs = {}
        #
        # for hidden_channels in [64, 32]:
        #     norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
        #     self.norms.append(norm)

    def forward(self, data):
        x = self.conv(data)
        return {"policy": x}


def compute_policy_loss(policy_logits, action_id, reward):

    cross_entropy = F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=action_id,
        reduction='none')

    loss = cross_entropy * reward
    loss = torch.mean(loss)

    return loss


def compute_reward(action, y_soft, y_truth):
    y_pred = torch.argmax(y_soft, dim=-1)
    pred_mask = (y_pred == y_truth).to(torch.int64)
    reward = (action == pred_mask).to(torch.float)
    # pos_reward = torch.nonzero(reward).view(-1)
    zero_index = reward - torch.ones_like(reward)
    # score = torch.stack([y_soft[i, y_pred[i]] for i in pos_reward])
    # reward[pos_reward] = score
    return reward + zero_index

def compute_reward_gmt(action, y_soft, device):
    # action_reward = torch.zeros(size=(action.shape[0], y_soft.shape[1])).int().to(device)
    # for i in range(0, action.shape[0]):
    #     action_reward[i][action[i]] = 1
    action_reward = torch.torch.nn.functional.one_hot(action, y_soft.shape[1])
    reward=action_reward+y_soft

    #Compute reward
    zero=torch.tensor([0.]).to(device)
    one = torch.tensor([1.]).to(device)
    tot_reward=torch.where(reward>1,one,zero)
    tot_reward=torch.sum(tot_reward,dim=1)
    # provide punishment to reward
    negative = torch.tensor([-5.]).to(device)
    tot_reward = torch.where(tot_reward == 0., negative, tot_reward)
    #Compute correct selection ratio
    one = torch.tensor([1.]).to(device)
    correct_reward=torch.where(reward>1,one,zero)
    correct_reward=torch.sum(correct_reward,dim=1)
    correct=torch.sum(correct_reward)
    reward_ratio=correct.item()/tot_reward.shape[0]

    return tot_reward,reward_ratio


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.mean(policy * log_policy)


def compute_value_loss(advantages):
    return 0.5 * torch.mean(advantages ** 2)
