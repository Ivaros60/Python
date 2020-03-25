# Run games and print corresponding curves
# ===========================

import numpy as np
import player
import arm
import matplotlib.pyplot as plt
from math import floor
from random import shuffle
from copy import deepcopy

# =============================
# tools
# =============================

def games(player, arms, nb_trials, nb_games):
    """
    Play one game and return stored informations

    :param player:
    :param arms:
    :param nb_trials:
    :param nb_games:
    :return:
    """
    best_mean = max([a.mean() for a in arms])
    chosen_arm = np.zeros((nb_games, nb_trials))
    reward = np.zeros((nb_games, nb_trials))
    expected_reward = np.zeros((nb_games, nb_trials))
    expected_best_reward = np.zeros((nb_games, nb_trials))
    for game in range(nb_games):
        player_for_one_game = deepcopy(player)
        for t in range(nb_trials):
            # play one turn
            i = player_for_one_game.choose_next_arm()
            # print("%d\t%f" % (i, arms[i].mean()))
            rew = arms[i].draw()
            player_for_one_game.update(i, rew)
            # store informations
            chosen_arm[game, t] = i
            reward[game, t] = rew
            expected_reward[game, t] = arms[i].mean()
            expected_best_reward[game, t] = best_mean
    return {'chosen_arm': chosen_arm, 'reward': reward, 'expected_reward': expected_reward, 'expected_best_reward': expected_best_reward}

def cumulative_reward(logs):
    """
    compute average cumulative reward
    :param logs:
    :return:
    """
    return np.mean(np.cumsum(logs['reward'], axis=1), axis=0)

def cumulative_regret(logs):
    """
    compute average cumulative regret
    :param logs:
    :return:
    """
    return np.mean(np.cumsum(logs['expected_best_reward']-logs['expected_reward'], axis=1), axis=0)

def nb_times_best__dist(logs):
    """
    compute average cumulative regret
    :param logs:
    :return:
    """
    return np.sum(logs['expected_best_reward'] == logs['expected_reward'], axis=1)

def logarithmic_indices(stop, n):
    """
    returns n indices logarithmically spanned from 0 to stop-1
    :param stop:
    :param n:
    :return:
    """
    return np.unique([floor(np.exp(i/(n-1)*np.log(stop)))-1 for i in range(n)])




if __name__ == '__main__':
    # =============================
    # play games
    # =============================
    arms = [arm.Bernoulli(p) for p in [0.2, 0.5]]
    nb_trials = 300
    nb_games = 100

    logs_oracle = games(player.Oracle(np.argmax([a.mean() for a in arms])), arms, nb_trials, nb_games)
    logs_ETC10 = games(player.ExploreThenCommit(len(arms), 10), arms, nb_trials, nb_games)
    logs_ETC40 = games(player.ExploreThenCommit(len(arms), 40), arms, nb_trials, nb_games)
    logs_ETC200 = games(player.ExploreThenCommit(len(arms), 200), arms, nb_trials, nb_games)

    # =============================
    # plot graphics
    # =============================

    # Cumulative Reward at time-step 300
    print("Cumulative Reward at time-step 300 (in average)")
    print("Oracle: \t", np.mean(np.sum(logs_oracle['reward'], axis=1)))
    print("ETC 10: \t", np.mean(np.sum(logs_ETC10['reward'], axis=1)))
    print("ETC 40: \t", np.mean(np.sum(logs_ETC40['reward'], axis=1)))
    print("ETC 200: \t", np.mean(np.sum(logs_ETC200['reward'], axis=1)))

    # - Cumulative reward = f(t) -
    plt.clf()
    inds = logarithmic_indices(nb_trials, 100)
    plt.plot(inds + 1, cumulative_reward(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_reward(logs_ETC10)[inds], "--", label='A/B, m = 10')
    plt.plot(inds + 1, cumulative_reward(logs_ETC40)[inds], "--", label='A/B, m = 40')
    plt.plot(inds + 1, cumulative_reward(logs_ETC200)[inds], "--", label='A/B, m = 200')
    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    # - Cumulative regret = f(t) -
    plt.clf()
    inds = logarithmic_indices(nb_trials, 100)
    plt.plot(inds + 1, cumulative_regret(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_regret(logs_ETC10)[inds], "--", label='A/B, m = 10')
    plt.plot(inds + 1, cumulative_regret(logs_ETC40)[inds], "--", label='A/B, m = 40')
    plt.plot(inds + 1, cumulative_regret(logs_ETC200)[inds], "--", label='A/B, m = 200')
    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Regret')
    plt.legend()
    plt.grid(True)
    plt.show()

    # - hist(Cumulative reward) -
    plt.clf()
    plt.hist(np.sum(logs_oracle['reward'], axis=1), 40, range=(0, 200), density=False, facecolor='r', alpha=0.75,
             label='Oracle')
    plt.hist(np.sum(logs_ETC10['reward'], axis=1), 40, range=(0, 200), density=False, facecolor='g', alpha=0.75,
             label='A/B, m = 10')
    plt.hist(np.sum(logs_ETC40['reward'], axis=1), 40, range=(0, 200), density=False, facecolor='b', alpha=0.75,
             label='A/B, m = 40')
    plt.xlabel('Cumulative Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # - hist(#time-steps optimal option is chosen) -
    plt.clf()
    plt.hist(nb_times_best__dist(logs_oracle), 40, range=(0, nb_trials), density=False, facecolor='r', alpha=0.75,
             label='Oracle')
    plt.hist(nb_times_best__dist(logs_ETC10), 40, range=(0, nb_trials), density=False, facecolor='g', alpha=0.75,
             label='A/B, m = 10')
    plt.hist(nb_times_best__dist(logs_ETC40), 40, range=(0, nb_trials), density=False, facecolor='b', alpha=0.75,
             label='A/B, m = 40')
    plt.xlabel('#time-steps')
    plt.ylabel('Frequency')
    # plt.title('Histogram of IQ')
    plt.legend()
    plt.grid(True)
    plt.show()

