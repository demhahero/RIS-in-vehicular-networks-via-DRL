# https://github.com/nikhilbarhate99/PPO-PyTorch
import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from Environment import Environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def main(item, control_bit):

    ############## Hyperparameters ##############
    # creating environment
    state_dim = Environment.observation_length
    action_dim = Environment.no_actions
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = N        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    env = Environment()

    env.M = item
    env.control_bit = control_bit

    env.initalize()

    # training loop
    for i_episode in range(1, max_episodes + 1):

        counter = 0
        state = env.reset()
        env.no_uav = False
        #state = np.asarray(state, dtype=np.float32)
        slot_rewards = 0
        done = False
        while not done:
            timestep += 1
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)


            reward, state, done = env.step(action=action)
            slot_rewards += reward



            state = np.asarray(state, dtype=np.float32)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward

            if done:
                DRL_bitrate.append(float("{:.3f}".format(env.total_reward)))
                DRL_jain.append(float("{:.3f}".format(env.jain)))
                break



        avg= 100
        if(i_episode % avg == 0):
            print("\n_________________________________________________________________________________________________")
            print("****Iteration=", i_episode, "\tBitrate=", float("{:.3f}".format(np.mean(DRL_bitrate[i_episode-avg:i_episode])))
                  , "\tJain=", float("{:.3f}".format(np.mean(DRL_jain[i_episode - avg:i_episode])))
                  )
            print("___________________________________________________________________________________________________")

    print(env.total_reward)
    del env

if __name__ == '__main__':

    To_DRL_bitrate = []
    To_DRL_jian = []

    items = [2]

    for x in items:

        DRL_bitrate = []
        DRL_jain = []


        N = 10000
        smoothing = 400


        main(item=100, control_bit=x)


        your_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = "results/result_" + str(your_datetime) + ".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(str(filename), "w")

        f.write("DRL_bitrate=" + str(DRL_bitrate) + "\n")
        f.write("DRL_jain=" + str(DRL_jain) + "\n")


        f.write("DRL_bitrate_avg=" + str(np.mean(DRL_bitrate[N-500:N])) + "\n")
        f.write("DRL_jain_avg=" + str(np.mean(DRL_jain[N-500:N])) + "\n")

        f.close()

        To_DRL_bitrate.append(np.mean(DRL_bitrate[N-500:N]))
        To_DRL_jian.append(np.mean(DRL_jain[N-500:N]))

        #rewards_smoothed = pd.Series(DRL_bitrate).rolling(smoothing, min_periods=0).mean()
        #plt.plot(rewards_smoothed, linewidth=1, color="b", label='Both')

    filename = "results/to"+"_result.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(str(filename), "w")
    f.write("To_DQN_bitrate=" + str(To_DRL_bitrate) + "\n")
    f.write("To_DQN_jain=" + str(To_DRL_jian) + "\n")
    print("END")
    f.close()
