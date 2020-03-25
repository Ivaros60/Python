from exp import games, cumulative_regret, cumulative_reward, logarithmic_indices
from random import shuffle
from player import Oracle, EpsilonNGreedy, ThompsonSamplingBernoulli, UCB1
from arm import MyBeta, Bernoulli
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# =============================
# play games
# =============================
environment = [MyBeta(mean=0.79), MyBeta(mean=0.17), MyBeta(mean=0.51), MyBeta(mean=0.45), MyBeta(mean=0.77), MyBeta(mean=0.38), MyBeta(mean=0.43), MyBeta(mean=0.43), MyBeta(mean=0.5), MyBeta(mean=0.41)]
shuffle(environment)
n_iter = 1000
n_games = 5

logs_oracle = games(Oracle(np.argmax([a.mean() for a in environment])), environment, n_iter, n_games)
logs_EG1 = games(EpsilonNGreedy(nb_arms=len(environment), c=1), environment, n_iter, n_games)
# XXX TO DO XXX       run also with c=10 and c=100
logs_TS = games(ThompsonSamplingBernoulli(nb_arms=len(environment), prior_s=0.5, prior_f=0.5), environment, n_iter, n_games)
logs_UCB1 = games(UCB1(nb_arms=len(environment)), environment, n_iter, n_games)



# =============================
# plot cumulative regret
# =============================
plt.clf()

inds = logarithmic_indices(n_iter, 100) # do not plot each point (too much with long runs)

plt.plot(inds + 1, cumulative_regret(logs_oracle)[inds], label='Oracle')
plt.plot(inds + 1, cumulative_regret(logs_EG1)[inds], "--", label='EG, c = 1')
# XXX TO DO XXX       plot res for c=10 and c=100
plt.plot(inds + 1, cumulative_regret(logs_TS)[inds], "--", label='TS')
plt.plot(inds + 1, cumulative_regret(logs_UCB1)[inds], "--", label='UCB1')
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
# XXX TO DO XXX       plot res for c=10 and c=100
plt.plot(inds + 1, cumulative_reward(logs_TS)[inds], "--", label='TS')
plt.plot(inds + 1, cumulative_reward(logs_UCB1)[inds], "--", label='UCB1')
plt.xlabel('Time')
plt.ylabel('Expected Cumulative Reward')
plt.legend()
plt.grid(True)
# plt.loglog()
plt.show()