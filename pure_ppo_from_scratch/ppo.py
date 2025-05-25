# note: gym needs to use numpy 1.25+ and not numpy 2.x
# pip install numpy==1.25.0
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
# pip install moviepy omegaconf matplotlib
# pip install gym==0.26.2
# pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
# pip install gym[classic_control] gym[atari] gym[accept-rom-license] gym[other]
import time
import random
import numpy as np
import matplotlib.pylab as plt
plt.style.use('dark_background')
from tqdm.notebook import tqdm
from omegaconf import DictConfig
import statistics

import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

from IPython.display import Video


seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


configs = {
    # experiment arguments
    "exp_name": "cartpole",
    "gym_id": "CartPole-v1", # the id of from OpenAI gym
    # training arguments
    "learning_rate": 1e-3, # the learning rate of the optimizer
    "total_timesteps": 1000000, # total timesteps of the training
    "max_grad_norm": 0.5, # the maximum norm allowed for the gradient
    # PPO parameters
    "num_trajcts": 32, # N
    "max_trajects_length": 64, # T
    "gamma": 0.99, # gamma
    "gae_lambda":0.95, # lambda for the generalized advantage estimation
    "num_minibatches": 2, # number of mibibatches used in each gradient
    "update_epochs": 2, # number of full rollout storage creations
    "clip_epsilon": 0.2, # the surrogate clipping coefficient
    "ent_coef": 0.01, # entroy coefficient controlling the exploration factor C2
    "vf_coef": 0.5, # value function controlling value estimation importance C1
    # visualization and print parameters
    "num_returns_to_average": 3, # how many episodes to use for printing average return
    "num_episodes_to_average": 23, # how many episodes to use for smoothing of the return diagram
    }

# batch_size is the size of the flatten sequences when trajcts are flatten
configs['batch_size'] = int(configs['num_trajcts'] * configs['max_trajects_length'])
# number of samples used in each gradient
configs['minibatch_size'] = int(configs['batch_size'] // configs['num_minibatches'])

configs = DictConfig(configs)

run_name = f"{configs.gym_id}__{configs.exp_name}__{seed}__{int(time.time())}"

# set up enviroment to interact with agent
# given action, the env is able to give observation (after doing the action), reward etc
# creating an env with random state
def make_env_func(gym_id, seed, idx, run_name, capture_video = False):
    def env_fun():
        env = gym.make(gym_id, render_mode = "rgb_gray")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
          # initiate the video capture if not already initiated
          if idx ==0:
            #wrapper to create the video of the performance
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    return env_fun

# create N (here is 32) parallel envs
envs = []
for i in range(configs.num_trajcts):
    envs.append(make_env_func(configs.gym_id, seed+i, i, run_name))
envs = gym.vector.SyncVectorEnv(envs)


# agent
# policy head: given a state, it returns the action, log_prob of actions, entropy and value score
# value head: given a state, it returens the estimated expected total reward/returns from this state
# policy head and value head are shared with the same foundational model (in this code, it is several FC blocks)
class FCBlock(nn.Module):
    """A generic fully connected residual block with good setup"""
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class Agent(nn.Module):
    """ an agent that creates actions and estimates values"""
    def __init__(self, env_observation_dim, action_space_dim, embed_dim = 64, num_blocks=2):
        super().__init__()
        # getting the observation and embed that into another space `embed_dim`
        self.embedding_layer = nn.Linear(env_observation_dim, embed_dim)
        # layers that are shared between policy head and value head
        # it not necessarily needed to have a shared layer, but here since that value and policy tasks are quite similar
        # we can use several shared layer to do multi-task learning
        self.shared_layers = nn.Sequential(*[FCBlock(embed_dim=embed_dim) for _ in range(num_blocks)])
        self.value_head = nn.Linear(embed_dim, 1)
        self.policy_head = nn.Linear(embed_dim, action_space_dim)
        # orthogonal initialization with a hi entropy for exploration at the start
        torch.nn.init.orthogonal_(self.policy_head.weight, 0.01)

    def value_func(self,state):
        hidden = self.shared_layers(self.embedding_layer(state))
        value = self.value_head(hidden)
        return value

    def policy(self, state, action=None):
        # state: (batch_size, num_states) where num_states = env_observation_dim
        # policy is supposed to create actions but here it takes actions as the input
        # this is for phase 2 of PPO where we want to analze the actions
        hidden = self.shared_layers(self.embedding_layer(state))
        logits = self.policy_head(hidden) # (batch_size, num_action)
        # Pytoch categorical class for sampling and probability calcucation
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # probs.log_prob(action): evaluating the likelihood of that action under the current policy
        # probs.entropy is a term used in the PPO loss to encourage exploration
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)

# Generalized Advantage Estimation (GAE)
# $Advantage (s, a)$ calculates how better or worse the return of taking the action $a$ at the state $s$ 
# is compared to expected return for all other actions in that state.
# we can approximate that with the below reverse formulas
# $δ_{t} = r_{t} + γV(s_{t+1}) - V(s_{t})$
# $\hat{A_{t}} = δ_{t} + γλ\hat{A}_{t+1}$
def gae(
    agent,
    cur_observations,   # the current state when advantages will be calculated, [num_trajcts, num_states]. Based on the rollout function next, the `cur_observations` is s_{t+1}
    rewards,            # rewards collected from trajectories of shape [num_trajcts, max_trajcts_length]
    dones,              # binary marker of end of trajectories of shape [num_trajcts, max_trajcts_length]
    values,             # value estimates collected over trajectories of shape [num_trajcts, max_trajcts_length]
):
    advantages = torch.zeros((configs.num_trajcts, configs.max_trajects_length))
    last_advantage = 0

    # the value after the last step (V_{t+1})
    with torch.no_grad():
        last_value = agent.value_func(cur_observations).reshape(1,-1) # (B, 1) -> (1, B)

    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(configs.max_trajects_length)):
        # below are the implementation of $δ_{t} = r_{t} + γV(s_{t+1}) - V(s_{t})$, $\hat{A_{t}} = δ_{t} + γλ\hat{A}_{t+1}$
        # mask if episode completed after step t
        mask = 1.0 - dones[:,t] # --> if we are looking for those trajectories that were ended quicker we don't need to any further calculation so we use the variable mask
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        delta = rewards[:,t] + configs.gamma * last_value - values[:,t]
        last_advantage = delta + configs.gamma * configs.gae_lambda * last_advantage
        advantages[:,t] = last_advantage
        last_value = values[:,t]

    advantages = advantages.to(device)
    returns = advantages + values
    return advantages, returns


def create_rollout(
    agent,
    envs,               # parallel envs creating trajectories
    cur_observation,    # starting observation of shape [num_trajcts, observation_dim], starting state is random initialized by envs.reset()
    cur_done,           # current termination status of shape [num_trajcts,]
    all_returns         # a list to track returns
):
    
    """
    rollout phase: create parallel trajectories and store them in the rolout storage
    """

    # cache empty tensors to store the rollouts
    observations = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                                 envs.single_observation_space.shape).to(device)
    actions = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                            envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    rewards = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    dones = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    values = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)

    for t in range(configs.max_trajects_length):
        observations[:,t] = cur_observation
        dones[:,t] = cur_done

        # give observation to the model and collect action, logprobs of actions, entropy and value
        with torch.no_grad():
            action, logprob, entropy, value = agent.policy(cur_observation)

        values[:,t] = value.flatten() # original shape is (configs.num_trajcts , 1) -> flat to (configs.num_trajcts)
        actions[:,t] = action
        logprobs[:,t] = logprob

        # apply the action to the env and collect observation and reward
        cur_observation, reward, cur_done, _, info = envs.step(action.cpu().numpy())
        rewards[:,t] = torch.Tensor(reward).to(device).view(-1) # reward is a python array, change it to tensor, view(-1) is not necessary
        cur_observation = torch.Tensor(cur_observation).to(device)
        cur_done = torch.Tensor(cur_done).to(device)

        # if an episode ended store its total reward for progress report
        if info:
            for item in info['final_info']:
                if item and "episode" in item.keys():
                    all_returns.append(item['episode']['r'])
                    break

    # create rollout storage
    rollout = {
        'cur_observation': cur_observation, # this is neeeded as it's s_{t+1} (not s_{t})
        'cur_done': cur_done,
        'observations': observations,
        'actions': actions,
        'logprobs' : logprobs,
        'values' : values,
        'dones' : dones,
        'rewards' : rewards
    }

    return rollout

# creating a standard pytorch dataset to store the RL model output in a standard dataset to be used for downstream tasks (NLP, LLM)
class Storage(Dataset):
    
    def __init__(self, rollout, advantages, returns, envs):
        # fill in the storage and flatten the parallel trajectories
        self.observations = rollout['observations'].reshape((-1,) + envs.single_observation_space.shape) # (num_traj * traj_length, 4)
        self.logprobs = rollout['logprobs'].reshape(-1) # (num_traj * traj_length)
        self.actions = rollout['actions'].reshape((-1,) + envs.single_action_space.shape).long() # (num_traj * traj_length, 2)
        self.advantages = advantages.reshape(-1) # (num_traj * traj_length)
        self.returns = returns.reshape(-1) # (num_traj * traj_length)


    def __getitem__(self, ix: int):
        item = [
            self.observations[ix],
            self.logprobs[ix],
            self.actions[ix],
            self.advantages[ix],
            self.returns[ix],
        ]
        return item

    def __len__(self) -> int:
        return len(self.observations)


def loss_clip(
    mb_oldlogprob,   # old logprob of mini batch actions collected during the rollout (old model)
    mb_newlogprob,   # new logprob of mini batch actions created by the new policy (new model)
    mb_advantages,   # mini batch of advantages collected during the rollout (old model)
):
    """
    policy loss with clipping to control gradients
    """
    ratio = torch.exp(mb_newlogprob - mb_oldlogprob)
    policy_loss = -mb_advantages * ratio          # since here we want to maximize the reward and pytorch optimizer always look for minimum value, we convert the sign by multiplying the negative -1
    # clipped policy gradient loss enforces closeness
    clipped_loss = -mb_advantages * torch.clamp(ratio, 1 - configs.clip_epsilon, 1 + configs.clip_epsilon)
    pessimistic_loss = torch.max(policy_loss, clipped_loss).mean()
    return pessimistic_loss # minimize this loss

def loss_vf(
    mb_oldreturns,    # mini batch of old returns collected during the rollout (old model)
    mb_newvalues,     # mini batch of values calculated by the new value function
):
    """
    enforcing the value function to give more accurate estimates of returns
    note: use old return as target rather than computing return on the fly to avoid moving target problem which would cause instability in trianing.
    """
    mb_newvalues = mb_newvalues.view(-1)
    loss = 0.5 * ((mb_newvalues - mb_oldreturns)**2).mean()
    return loss # minimize this loss


agent = Agent(
    env_observation_dim=envs.single_observation_space.shape[0],
    action_space_dim=envs.single_action_space.n
).to(device)

optimizer = optim.Adam(agent.parameters(), lr=configs.learning_rate)

def train():
    # track returns
    all_returns = []
    # initialize the game
    # evns.reset()[0] has shape: (number of envs in this case num_trajcts, number of states)
    cur_observation = torch.Tensor(envs.reset()[0]).to(device)
    cur_done = torch.zeros(configs.num_trajcts).to(device)

    num_updates = configs.total_timesteps // configs.batch_size

    for update in range(1, num_updates + 1):

        ##############################################
        # Phase 1: rollout creation

        # parallel envs creating trajectories
        rollout = create_rollout(agent, envs, cur_observation, cur_done, all_returns)

        cur_done = rollout['cur_done']
        cur_observation = rollout['cur_observation'] # observation at t+1
        rewards = rollout['rewards']
        dones = rollout['dones']
        values = rollout['values']

        # calculating advantages
        advantages, returns = gae(agent, cur_observation, rewards, dones, values)

        # a dataset containing the rollouts
        dataset = Storage(rollout, advantages, returns, envs)

        # a standard dataloader made out of current storage
        trainloader = DataLoader(dataset, batch_size=configs.minibatch_size, shuffle=True)


        ##############################################
        # Phase 2: model update

        # linearly shrink the lr from the initial lr to zero
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * configs.learning_rate

        # training loop
        for epoch in range(configs.update_epochs):
            for batch in trainloader:
                mb_observations, mb_logprobs, mb_actions, mb_advantages, mb_returns = batch

                # we calculate the distribution of actions through the updated model revisiting the old trajectories
                _, mb_newlogprob, mb_entropy, mb_newvalues = agent.policy(mb_observations, mb_actions)

                policy_loss = loss_clip(mb_logprobs, mb_newlogprob, mb_advantages)

                value_loss = loss_vf(mb_returns, mb_newvalues)

                # average entory of the action space
                entropy_loss = mb_entropy.mean() # maximize entropy so minimize -1 * entropy

                # full weighted loss
                loss = policy_loss - configs.ent_coef * entropy_loss + configs.vf_coef * value_loss

                optimizer.zero_grad()
                # this will update parameters of agent because we use mb_newlogprob, mb_entropy, mb_newvalues in the loss
                loss.backward()

                # extra clipping of the gradients to avoid overshoots
                # gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), configs.max_grad_norm)
                optimizer.step()

    envs.close()

    mean_value = statistics.mean(all_returns)
    std_value = statistics.stdev(all_returns)  # Use pstdev() for population std dev
    max_value = max(all_returns)
    print("Mean:", mean_value)
    print("Standard Deviation:", std_value)
    print("Max:", max_value)
    # ploting the max return should see a generally increasing curve
    # import matplotlib.pyplot as plt
    # plt.plot(range(len(all_returns)), all_returns)
    # # Labels and title
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Index vs Value')
    # plt.grid(True)
    # # Show the plot
    # plt.show()
    return all_returns

def main():
    train()

if __name__ == "__main__":
    main()

