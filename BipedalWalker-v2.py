import gym
from gym.wrappers import Monitor
import math
import random
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import base64

import ReplayMemory
import OUnoise
import ActorNet
import CriticNet

from matplotlib import animation

env = gym.make("BipedalWalker-v2")
env._max_episode_steps = 1000

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env.reset()

BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.00
EPS_END = 0.00
EPS_DECAY = 2000
WEIGHT_DECAY = 0.001
TARGET_UPDATE = 5
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
MEMORY_SIZE = 1000000
EPISODE_SIZE = 10000
TAU = 0.001
ou_noise_theta = 0.4
ou_noise_sigma = 0.2

REWARD_WEIGHT = 1.0
RECORD_INTERVAL = 100
RENDER_INTERVAL = 1

# gym 행동 공간에서 행동의 숫자를 얻습니다.
# n_actions = env.action_space.n
n_actions = env.action_space.shape[0]
n_obvs = env.observation_space.shape[0]

actor = ActorNet.Actor(n_obvs, n_actions).to(device)
# actor.eval()
actor_target = ActorNet.Actor(n_obvs, n_actions).to(device)
actor_target.load_state_dict(actor.state_dict())
# actor_target.eval()

critic = CriticNet.Critic(n_obvs, n_actions).to(device)
# critic.eval()
critic_target = CriticNet.Critic(n_obvs, n_actions).to(device)
critic_target.load_state_dict(critic.state_dict())
# critic_target.eval()

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
memory = ReplayMemory.ReplayMemory(n_actions, n_obvs, MEMORY_SIZE, BATCH_SIZE)

noise = OUnoise.OUNoise(
    n_actions,
    theta=ou_noise_theta,
    sigma=ou_noise_sigma,
)

steps_done = 0


def select_action(state, steps_done=None):
    # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    state = torch.FloatTensor(state).unsqueeze(0)
    if sample < eps_threshold:
        # selected_action = [np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)]
        selected_action = np.random.uniform(-1, 1, n_actions)
    else:
        actor.eval()
        with torch.no_grad():
            selected_action = actor(
                state.to(device)
            )[0].detach().cpu().numpy()

    _noise = noise.sample()
    actor.train()
    for act in selected_action:
        act = np.clip(act + _noise, -1.0, 1.0)
    return selected_action


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def target_soft_update():
    # Soft-update: target = tau*local + (1-tau)*target
    tau = TAU

    for t_param, l_param in zip(
            actor_target.parameters(), actor.parameters()
    ):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    for t_param, l_param in zip(
            critic_target.parameters(), critic.parameters()
    ):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0, 0
    samples = memory.sample_batch()
    state = torch.FloatTensor(samples["obs"]).to(device)
    next_state = torch.FloatTensor(samples["next_obs"]).to(device)
    action = torch.FloatTensor(samples["acts"]).to(device)
    reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    masks = 1 - done
    next_action = actor_target(next_state)
    next_value = critic_target(next_state, next_action)
    curr_return = reward + (GAMMA * next_value * masks)

    # train critic
    values = critic(state, action)
    critic_loss = F.mse_loss(values, curr_return)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # train actor

    loss = critic(state, actor(state))
    actor_loss = -loss.mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # target update
    target_soft_update()

    return actor_loss.data, critic_loss.data


frames = []
batch_count = 0
for i_episode in range(EPISODE_SIZE):
    # 환경과 상태 초기화
    obv = env.reset()
    total_actor_loss = 0
    total_critic_loss = 0
    total_reward = 0
    top_reward = -1
    total_action_count = [0, 0, 0]

    for t in count():
        if i_episode % RENDER_INTERVAL == 0:
            frame = env.render(mode="rgb_array")
            if i_episode % RECORD_INTERVAL == 0:
                frames.append(frame)
        # 행동 선택과 수행
        action = select_action(obv, steps_done)
        steps_done += 1
        next_obv, reward, done, _ = env.step(action)
        if reward > top_reward:
            top_reward = reward
        total_reward += reward
        #reward -= 0.01
        reward = REWARD_WEIGHT * reward
        # 메모리에 변이 저장
        assert obv is not None
        memory.store(obv, action, reward, next_obv, done)

        # 다음 상태로 이동
        obv = next_obv

        # 최적화 한단계 수행(목표 네트워크에서)
        actor_loss, critic_loss = optimize_model()
        total_actor_loss += actor_loss
        total_critic_loss += critic_loss

        if done:
            E = eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * steps_done / EPS_DECAY)
            episode_durations.append(total_reward)
            print(
                '%d episode , %d step , %.2f Actor Loss, %.2f Critic Loss,  %.2f Threshold , %.2f Top reward, %.2f Total reward' \
                % (
                    i_episode, t + 1, total_actor_loss / (t + 1), total_critic_loss / (t + 1), E, top_reward,
                    total_reward))
            # print(total_action_count)
            plot_durations()
            total_actor_loss = 0
            total_critic_loss = 0
            top_reward = 0
            total_reward = 0
            total_action_count = [0, 0, 0]
            break
    # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    #if i_episode % TARGET_UPDATE == 0:
         #target_soft_update()
print('Complete')
env.close()
plt.show()

# Imports specifically so we can render outputs in Colab.
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
fig = plt.figure()


def display_frames_as_gif(frame):
    """Displays a list of frames as a gif, with controls."""
    patch = plt.imshow(frame[0].astype(int))

    def animate(i):
        patch.set_data(frame[i].astype(int))

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=0.1, blit=False
    )
    # display(display_animation(anim, default_mode='loop'))
    # Set up formatting for the movie files
    # display(HTML(data=anim.to_html5_video()))
    FFwriter = animation.FFMpegWriter(fps=30)
    anim.save('./Result_Animations.mp4', writer=FFwriter)
    # show_video()


# display
display_frames_as_gif(frames)
print('done')
