import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from customer import DEFAULT_SIM
import matplotlib.pyplot as plt


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=2,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SAC_DP",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="dynamic-pricing",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="DP",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(2e5),
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk

class ReplayBuffer:
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_memory = np.zeros((self.mem_size, input_shape))
    self.new_state_memory = np.zeros((self.mem_size, input_shape))
    self.action_memory = np.zeros((self.mem_size, n_actions))
    self.reward_memory = np.zeros(self.mem_size)
    # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

  def store_transition(self, state, action, reward, new_state):
    index = self.mem_cntr %self.mem_size

    self.state_memory[index] = state
    self.new_state_memory[index] = new_state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    # self.terminal_memory[index] = done

    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)

    batch = np.random.choice(max_mem, batch_size, replace = False)

    states = self.state_memory[batch]
    states_ = self.new_state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    # dones = self.terminal_memory[batch]
    
    states = torch.from_numpy(states).cuda()
    states = states.to(torch.float32)
    states_ = torch.from_numpy(states_).cuda()
    states_ = states_.to(torch.float32)
    actions = torch.from_numpy(actions).cuda()
    actions = actions.to(torch.float32) 
    rewards = torch.from_numpy(rewards).cuda()
    rewards = rewards.to(torch.float32)

    return states, actions, rewards, states_


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(np.array([5]).prod() + np.prod([1]), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(np.array([5]).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod([1]))
        self.fc_logstd = nn.Linear(256, np.prod([1]))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((15.0 - 5.0) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((5.0 + 15.0) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(15.0)

    actor = Actor().to(device)
    qf1 = SoftQNetwork().to(device)
    qf2 = SoftQNetwork().to(device)
    qf1_target = SoftQNetwork().to(device)
    qf2_target = SoftQNetwork().to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor([1]).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        5,
        1,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    comp_price_initial = 10.0
    d = 0.5
    quality_self = 4
    quality_comp = 3
    rating_self = 0.75
    rating_comp = 0.65
    np.random.seed(args.seed)
    
    rewards_tracker = []
    cumulative_rewards = []
    cnt = []
    
    obs = [comp_price_initial, quality_comp, rating_comp, quality_self, rating_self]
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = random.sample(range(5, 15), 1)
            actions = np.array(actions)
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, dones, infos = envs.step(actions)
        # next_obs = [comp_price]
        next_comp_price = actions[0] - d
        if next_comp_price <= 7.5:
            next_comp_price = 12.0
            
        next_obs = [next_comp_price, quality_comp, rating_comp, quality_self, rating_self]
        
        value_self = actions[0] + np.random.uniform(0, 1) * quality_self + np.random.uniform(0, 0.5) * (1 - rating_self)
        value_comp = obs[0] + np.random.uniform(0, 1) * quality_comp + np.random.uniform(0, 0.5) * (1 - rating_comp)
        # rewards = DEFAULT_SIM(0.5, actions[0], comp_price, 10.0) * actions[0]
        if value_self <= value_comp:
            rewards = actions[0]
        else:
            rewards = 0.0
            
        cumulative_rewards.append(rewards)
        if global_step % 5000 == 0:
            print(f"Global Steps: {global_step} --- Rewards: {rewards} -- Price: {actions[0]}")
            rewards_tracker.append(np.mean(cumulative_rewards))
            cumulative_rewards = []
            cnt.append(global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # for info in infos:
        #     if "episode" in info.keys():
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # for idx, d in enumerate(dones):
        #     if d:
        #         real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.store_transition(obs, actions, rewards, real_next_obs)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            observations, actions, rewards, next_observations = rb.sample_buffer(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations)
                qf1_next_target = qf1_target(next_observations, next_state_actions)
                qf2_next_target = qf2_target(next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = rewards.flatten() + args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(observations, actions).view(-1)
            qf2_a_values = qf2(observations, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(observations)
                    qf1_pi = qf1(observations, pi)
                    qf2_pi = qf2(observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                wandb.log(
                    {"qf1_values": qf1_a_values.mean().item(), "qf2_values": qf2_a_values.mean().item(), "qf1_loss": qf1_loss.item(), "qf2_loss": qf2_loss.item(), "qf_loss": qf_loss.item() / 2.0,
                     "actor_loss": actor_loss.item(), "alpha": alpha, "alpha_loss": alpha_loss.item(), "Rewards": rewards},
                    step = global_step
                )
                
    plt.plot(cnt, rewards_tracker, marker = 'o', linewidth = 1.5, color = 'r')
    plt.title('Mean Rewards')
    plt.xlabel('Global Step')
    plt.ylabel('Rewards')
    plt.show()

    writer.close()