import numpy as np

# functions for Q learning of differnet orders
# Q_learning_X_order(seq, lr, beta), where lr and beta are parameters of Q leaning agent and seq 
# is input sequence on which agent learns
# output is Q function in tabular form, and rewards, actions and actions' probabilities during trial

def softmax(x, b = 10):
    scoreMatExp = np.exp((np.asarray(x) - np.max(x)) * b)
    res = scoreMatExp / scoreMatExp.sum(0)
    res[np.isnan(res)] = 1
    return res

def Q_learning_zero_order(seq, lr, beta,):
    Q = np.array([0.1, 0.1])
    rewards = []
    actions = [np.random.randint(0, 2)]
    probs = []
    for s in seq[1:]:
        prob = softmax(Q)
        a = np.random.choice([0, 1], p = prob)
        probs.append(prob)
        actions.append(a)
        reward = 1 if a == s else 0
        rewards.append(reward)
        Q[a] += lr * (reward - Q[a])
        Q[1 - a] += lr * (1 - reward - Q[a])
    return Q, rewards, actions, probs


def Q_learning_first_order(seq, lr, beta,):
    Q = np.array([[0., 0.], [0., 0.]])
    prev_state = seq[0]
    rewards = []
    actions = [np.random.randint(0, 2)]
    probs = []
    for s in seq[1:]:
        prob = softmax(Q[prev_state])
        a = np.random.choice([0, 1], p = prob)
        probs.append(prob)
        actions.append(a)
        reward = 1 if a == s else 0
        rewards.append(reward)
        Q[prev_state, a] += lr * (reward - Q[prev_state, a])
        Q[prev_state, 1 - a] += lr * (1 - reward - Q[prev_state, a])
        prev_state = s
    return Q, rewards, actions, probs


def Q_learning_second_order(seq, lr, beta,):
    Q = np.zeros((2, 2, 2))
    prev_state = seq[0]
    prev_prev_state = np.random.randint(0, 2)
    rewards = []
    actions = [np.random.randint(0, 2)]
    probs = []
    for s in seq[1:]:
        prob = softmax(Q[prev_state, prev_prev_state])
        a = np.random.choice([0, 1], p = prob)
        probs.append(prob)
        actions.append(a)
        reward = 1 if a == s else 0
        rewards.append(reward)
        Q[prev_state, prev_prev_state, a] += lr * (reward - Q[prev_state, prev_prev_state, a])
        Q[prev_state, prev_prev_state, 1 - a] += lr * (1 - reward - Q[prev_state, prev_prev_state, a])
        prev_prev_state, prev_state = prev_state, s        
    return Q, rewards, actions, probs