import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt

#Fonctions permettant de simuler le jeu 
from exp import games, cumulative_regret, cumulative_reward, logarithmic_indices

#Fonctions permettant de simuler le joueur 
from player import Oracle, EpsilonNGreedy, ThompsonSamplingBernoulli, UCB1


def OptimizeEN(environment,n_iter,n_games) :

    # =============================
    # player
    # =============================

    logs_oracle = games(Oracle(np.argmax([a.mean() for a in environment])), environment, n_iter, n_games)
    logs_EG1 = games(EpsilonNGreedy(nb_arms=len(environment), c=1), environment, n_iter, n_games)
    logs_EG10 = games(EpsilonNGreedy(nb_arms=len(environment), c=10), environment, n_iter, n_games)
    logs_EG50 = games(EpsilonNGreedy(nb_arms=len(environment), c=50), environment, n_iter, n_games)
    logs_EG100 = games(EpsilonNGreedy(nb_arms=len(environment), c=100), environment, n_iter, n_games)


    # =============================
    # plot cumulative regret
    # =============================
    plt.clf()

    inds = logarithmic_indices(n_iter, 100) # do not plot each point (too much with long runs)

    plt.plot(inds + 1, cumulative_regret(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_regret(logs_EG1)[inds], "--", label='EG, c = 1')
    plt.plot(inds + 1, cumulative_regret(logs_EG10)[inds], "--", label='EG, c = 10')
    plt.plot(inds + 1, cumulative_regret(logs_EG50)[inds], "--", label='EG, c = 50')
    plt.plot(inds + 1, cumulative_regret(logs_EG100)[inds], "--", label='EG, c = 100')

    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Regret')
    plt.legend()
    plt.grid(True)
    # plt.loglog()
    plt.show()

    # =============================
    # plot cumulative reward
    # =============================
    plt.clf()

    inds = logarithmic_indices(n_iter, 100) # do not plot each point (too much with long runs)

    plt.plot(inds + 1, cumulative_reward(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_reward(logs_EG1)[inds], "--", label='EG, c = 1')
    plt.plot(inds + 1, cumulative_reward(logs_EG10)[inds], "--", label='EG, c = 10')
    plt.plot(inds + 1, cumulative_reward(logs_EG50)[inds], "--", label='EG, c = 50')
    plt.plot(inds + 1, cumulative_reward(logs_EG100)[inds], "--", label='EG, c = 100')

    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Reward')
    plt.legend()
    plt.grid(True)
    # plt.loglog()
    plt.show()
    return "End"

def OptimizeUCB1(environment, n_iter, n_games) :

    logs_oracle = games(Oracle(np.argmax([a.mean() for a in environment])), environment, n_iter, n_games)

    logs_UCB1 = games(UCB1(nb_arms=len(environment),alpha = 1), environment, n_iter, n_games)
    logs_UCB05 = games(UCB1(nb_arms=len(environment),alpha = 0.5), environment, n_iter, n_games)
    logs_UCB03 = games(UCB1(nb_arms=len(environment),alpha = 0.3), environment, n_iter, n_games)
    logs_UCB15 = games(UCB1(nb_arms=len(environment),alpha = 1.5), environment, n_iter, n_games)

    # =============================
    # plot cumulative regret
    # =============================
    plt.clf()

    inds = logarithmic_indices(n_iter, 100) # do not plot each point (too much with long runs)

    plt.plot(inds + 1, cumulative_regret(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_regret(logs_UCB1)[inds], "--", label='EG, alpha = 1')
    plt.plot(inds + 1, cumulative_regret(logs_UCB05)[inds], "--", label='EG, alpha = 0.5')
    plt.plot(inds + 1, cumulative_regret(logs_UCB03)[inds], "--", label='EG, alpha = 0.3')
    plt.plot(inds + 1, cumulative_regret(logs_UCB15)[inds], "--", label='EG, alpha = 1.5')

    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Regret')
    plt.legend()
    plt.grid(True)
    # plt.loglog()
    plt.show()

    # =============================
    # plot cumulative reward
    # =============================
    plt.clf()

    inds = logarithmic_indices(n_iter, 100) # do not plot each point (too much with long runs)

    plt.plot(inds + 1, cumulative_reward(logs_oracle)[inds], label='Oracle')
    plt.plot(inds + 1, cumulative_reward(logs_UCB1)[inds], "--", label='EG, alpha = 1')
    plt.plot(inds + 1, cumulative_reward(logs_UCB05)[inds], "--", label='EG, alpha = 0.5')
    plt.plot(inds + 1, cumulative_reward(logs_UCB03)[inds], "--", label='EG, alpha = 0.3')
    plt.plot(inds + 1, cumulative_reward(logs_UCB15)[inds], "--", label='EG, alpha = 1.5')
    plt.xlabel('Time')
    plt.ylabel('Expected Cumulative Reward')
    plt.legend()
    plt.grid(True)
    # plt.loglog()
    plt.show()
    return "End"


