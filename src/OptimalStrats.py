import numpy as np
from bandits import *
# functions that perform optimal decision making strategy
# depending on preset order, prediction of stimulus is based on the amount of x-order switches and stays

def compute_optimal_curve(P, order, bandit_order = None, N_ep = 5000, trial_len = 80):
    if bandit_order is None:
        bandit_order = order
    if order == 'zero':
        func = optimal_strat_zero

    elif order == 'first':
        func = optimal_strat_first

    elif order == 'second':
        func = optimal_strat_second
    else:
        print("Wrong order for compute_optimal_curve : % s" % order)
    
    if bandit_order == 'zero':
        bandit = zero_order_bandit(P = P)
    elif bandit_order == 'first':
        bandit = first_order_bandit(P = P)
    elif bandit_order == 'second':
        bandit = second_order_bandit(P = P)
    else:
        print("Wrong bandit order for compute_optimal_curve : % s" % bandit_order)
               
        
    mean_reward_hist = []
    for i in range(N_ep):
        states = []
        bandit.reset()
        for i in range(trial_len):
            states.append(bandit.reward_pos)
            bandit.act(1)
        rewards, actions = func(states)
        mean_r = np.mean(rewards)
        mean_reward_hist.append(mean_r)
    return np.mean(mean_reward_hist)

def optimal_strat_zero(states,):
    c0 = 0
    c1 = 0
    rewards = []
    actions = []
    for i in range(0, len(states)):
        s = states[i]
        if c0  > c1:
            a = 0
        elif c0 < c1:
            a = 1
        else:
            a = np.random.randint(0, 2)
        r = 1 if s == a else 0
        rewards.append(r)
        actions.append(a)
        if s == 1:
            c1 += 1
        else:
            c0 += 1
    return rewards, actions

def optimal_strat_first(states,):
    stays_count = 0
    switches_count = 0
    actions, rewards = [], []
    a = np.random.randint(0, 2)
    r = 1 if states[0] == a else 0
    rewards.append(r)
    actions.append(a)
    prev_s = states[0]
    for i in range(1, len(states)):
        s = states[i]
        if stays_count  > switches_count:
            a = prev_s
        elif stays_count < switches_count:
            a = 1 - prev_s
        else:
            a = np.random.randint(0, 2)
        r = 1 if s == a else 0
        rewards.append(r)
        actions.append(a)
        if s == prev_s:
            stays_count += 1
        else:
            switches_count +=1
        prev_s = s
    return rewards, actions


def optimal_strat_second(states,):
    stays_count = 0
    switches_count = 0
    actions, rewards = [], []
    a = np.random.randint(0, 2)
    r = 1 if states[0] == a else 0
    rewards.append(r)
    actions.append(a)
    prev_s = states[0]
    prev_prev_s = states[1]
    for i in range(2, len(states)):
        s = states[i]
        if stays_count  > switches_count:
            a = prev_prev_s
        elif stays_count < switches_count:
            a = 1 - prev_prev_s
        else:
            a = np.random.randint(0, 2)
        r = 1 if s == a else 0
        rewards.append(r)
        actions.append(a)
        if s == prev_prev_s:
            stays_count += 1
        else:
            switches_count +=1
        prev_prev_s = prev_s
        prev_s = s
    return rewards, actions