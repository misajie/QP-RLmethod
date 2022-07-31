import os
import time
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import List


def run_episode(env, agent):
    state = env.reset()
    while True:
        action = agent.select_action(state)
        # print("state:",np.where(state[:,-1]>0))
        # print(action)
        # print(agent.env.cur_episode.solver.WorkingSet)
        # action = env.random_action(state)  # test score:-6.61, average niter:13.37
        # action = env.as_action(state)    # AS policy:test score:-5.94, average niter:12.7
        nstate, reward, done,_ = env.step(action)
        state = nstate
        if done:
            break
    return env.cur_episode.score, env.cur_episode.niter


def explore(agent, env, init_episode, save_dir):
    agent.is_test = True
    init_scores = 0
    st = time.time()
    memory_path = save_dir + "/memory.pkl"
    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            agent.memory = pickle.load(f)
    else:
        for _ in range(init_episode):
            state = env.reset()
            while True:
                action = env.random_action(state)
                nstate, reward, done, _ = env.step(action)
                transition = dict(
                    state=state,
                    action=action,
                    nstate=nstate,
                    reward=reward,
                    done=done
                )
                # print(transition)
                agent.memory.add(**transition)
                state = nstate
                if done:
                    break
            init_scores += env.cur_episode.niter
        with open(memory_path, "wb") as f:
            pickle.dump(agent.memory, f)
        ne = time.time()
        print("init scores:", init_scores / init_episode, ne - st)


def train(env, test_env, agent, epoch, init_episode, save_dir):
    # warm replay memory
    explore(agent, env, init_episode, save_dir)
    print("epoch | score | q_loss | ac_loss | time")
    critic_lossls, actor_lossls, scores = [], [], []
    best_score = -1000
    start = time.time()
    agent.is_test = False
    for i in range(10):
        critic_loss, actor_loss = agent.update()
        print(i,critic_loss)
    agent.save_or_load_agent(save_dir, if_save=True)
    agent.save_or_load_agent(save_dir, if_save=False)
    for i in range(epoch):
        c_loss, a_loss = [], []
        state = env.reset()
        while True:
            action = agent.select_action(state)
            nstate, reward, done, _ = env.step(action)
            transition = dict(
                state=state,
                action=action,
                nstate=nstate,
                reward=reward,
                done=done
            )
            # print(transition)
            agent.memory.add(**transition)
            state = nstate
            if done:
                break

        critic_loss, actor_loss = agent.update()
        c_loss.append(critic_loss)
        a_loss.append(actor_loss)
        # need to change for different agent
        with torch.no_grad():
            critic_lossls.append(np.mean(c_loss))
            actor_lossls.append(np.mean(a_loss))

        scores.append(env.cur_episode.score)
        # print the train and test performance, can plot here
        check_freq = 50
        if i % check_freq == 0 or i == epoch - 1:
            end = time.time()
            with torch.no_grad():  # draw without considering separate reward
                print(i, np.mean(scores[-check_freq:]),
                      np.mean(critic_lossls[-check_freq:]),
                      np.mean(actor_lossls[-check_freq:]), end - start)
                plot_loss(i, scores, critic_lossls, actor_lossls)
            start = time.time()

        if (i + 1) % 200 == 0 or i == epoch - 1:
            print(i, "testing performance:")
            cur_score = test(test_env, agent, 100)
            if cur_score > best_score:
                best_score = cur_score
                agent.save_or_load_agent(save_dir, if_save=True)


def test(env, agent, test_no):
    start = time.time()
    agent.is_test = True
    with torch.no_grad():
        scores, niters, results = [], [], []
        for i in range(test_no):
            score, niter = run_episode(env, agent)
            scores.append(score)
            niters.append(niter)
    end = time.time()
    print("test score:{}, average niter:{},time:{}".format(np.mean(scores), np.mean(niters), end - start))
    return np.mean(scores)


def plot_loss(
        frame_idx: int,
        scores: List[float],
        critic_losses: List[float],
        actor_losses: List[float]
):
    """Plot the training progresses."""

    def subplot(loc: int, title: str, values: List[float]):
        plt.subplot(loc)
        plt.title(title)
        plt.plot(values)

    subplot_params = [
        (221, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
        (222, "critic_loss", critic_losses),
        (223, "ac_loss", actor_losses),
    ]

    clear_output(True)
    plt.figure(figsize=(30, 5))
    for loc, title, values in subplot_params:
        subplot(loc, title, values)
    plt.show()
