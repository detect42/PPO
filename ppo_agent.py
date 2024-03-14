import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal

def orthogonal_init(layer,gain=1.0):
    """
    通常情况下，如果不指定gain参数，默认值为1.0，即不进行缩放。但有时你可能想要尝试不同的缩放因子，以查看对模型性能的影响。例如，如果你的激活函数是Sigmoid或Tanh，你可能会尝试使用较小的gain值，因为这些激活函数在输入较大时会饱和。反之，如果你的激活函数是ReLU，你可能会尝试使用较大的gain值，以确保初始化的权重不会使大部分激活神经元变为零。
    """
    nn.init.orthogonal_(layer.weight,gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Beta(nn.Module):
    def __init__(self,args):
        super(Actor_Beta,self).__init__()
        self.fc1 = nn.Linear(args.state_dim,args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width,args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width,args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width,args.action_dim)
        self.activate_func = [nn.ReLU(),nn.Tanh()][args.use_tanh] # choose specific activation function

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

        def forward(self,s):
            s = self.activate_func(self.fc1(s))
            s = self.activate_func(self.fc2(s))
            alpha = F.softplus(self.alpha_layer(s)) + 1
            beta = F.softplus(self.beta_layer(s)) + 1
            return alpha,beta

        def get_dist(self,s):
            alpha,beta = self.forward(s)
            dist = Beta(alpha,beta)
            return dist

        def mean(self,s):
            alpha,beta = self.forward(s)
            mean = alpha/(alpha+beta) # The mean of the beta distribution
            return mean

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  #! exist batchsize as the first dimension. To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

class PPO_continuous_agent():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self,s):
        s = torch.unsqueeze(torch.FloatTensor(s),0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self,s):
        s = torch.unsqueeze(torch.FloatTensor(s),0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s) #! python provide the distribution class of Beta
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a = torch.clamp(a,-self.max_action,self.max_action)
                a_logprob = dist.log_prob(a)
        return a.numpy().flatten(),a_logprob.numpy().flatten()

    def update(self,replay_buffer,total_steps):
        s,a,a_logprob,r,s_,dw,done = replay_buffer.numpy_to_tensor()
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the   max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = [] #! Advantage
        gae = 0
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0-dw) * vs_ - vs
            for delta,d in zip(reversed(deltas.flatten().numpy()),reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * (1.0-d) * gae
                adv.insert(0,gae) # Insert the element at the beginning of the list
            adv = torch.tensor(adv,dtype=torch.float).view(-1,1)
            v_targets = adv + vs
            if self.use_adv_norm: # Trick 1:advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)),self.mini_batch_size,False):#! ---------
                dist_now = self.actor.get_dist(s[index])
               # print("!!,",dist_now.entropy().shape)
                dist_entropy = dist_now.entropy().sum(1,keepdim=True) #! 熵本来就是一维，为什么还要求和(因为有6个action dim，每一个dim里都是一个分布，所以熵有六个，要加起来)
              #  print(dist_entropy.shape)
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1,surr2) - self.entropy_coef * dist_entropy
                #! 我一直觉得这里取min没有道理
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()  #? min and clip operation can be derivate ?
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_targets[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self,total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now




