import numpy as np

class Bandit:
    def __init__(self, k_arm=10, epsilon=0.0, optimistic_init=0.0, step_size=0.1, 
                 sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, 
                 true_reward=0.0):
        """
        Args:
            k_arm (int): # of arms.
            epsilon (float): the probability for exploration in epsilon-greedy algorithm.
            optimistic_init (float): init the estimation for each action.
            step_size (float): constant step size for updating estimations.
            
            sample_averages (bool): if True, use sample averages to update estimations instead of constant step size.
            UCB_param (float): if not None, use UCB algorithm to select action.
            gradient (bool): if True, use gradient based bandit algorithm.
            gradient_baseline (bool): if True, use average reward as baseline for gradient based bandit algorithm.
   
            true_reward (float):
        """
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = optimistic_init
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        

    def reset(self):
        """Reset the k-armed bandits before starting the simulation.
        """
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)


    def act(self):
        """Perform the action selection.
        """
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # UCB
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                     self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        # gradient
        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        # greedy
        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])


    def step(self, action):
        """Take the step, get the reward and update the q_estimation.
        """
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        # update estimation
        if self.sample_averages:
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation = self.q_estimation + self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

        return reward
