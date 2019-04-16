import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bandit import Bandit
import os


SAVING_PATH = "../../images/chapter02"


def simulate(bandits, runs, time):
    """Start the simulation and averade the rewards with several runs.
    Args:
        bandits ([Bandit]): a list of bandits to be tested.
        runs (int): test each bandit with the # of runs.
        time (int): the # of the time-steps in each run.
    """
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)

    for i, bandit in enumerate(bandits):
        print("Bandit: {}/{}".format(i+1, len(bandits)))
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    best_action_counts = best_action_counts.mean(axis=1)
    rewards = rewards.mean(axis=1)
    return best_action_counts, rewards


def reward_distribution(k=10):
    """Create the stationatry k-armed bandit.
    Args:
        k (int): # of arms.
    """
    distributions = np.random.randn(200, k)
    offsets = np.random.randn(k)
    plt.violinplot(dataset=distributions + offsets)

    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig(os.path.join(
        SAVING_PATH, "figure_2_1_reward_distribution.png"))
    plt.close()


def epsilon_greedy(runs=2000, time=1000):
    """Test the k-armed bandit with the policy of epsilon greedy. 
    Args:
        runs (int): test each bandit with the # of runs.
        time (int): the # of the time-steps in each run.
    """
    epsilons = [0, 0.01, 0.1, 0.2]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]

    print("===== {} =====".format("Epsilon greedy"))
    best_action_counts, rewards = simulate(bandits, runs, time)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label="epsilon = %.02f" % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label="epsilon = %.02f" % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig(os.path.join(SAVING_PATH, "figure_2_2_epsilon_greedy.png"))
    plt.close()


def optimistic_initial(runs=2000, time=1000):
    """Test the k-armed bandit with optimistic initialization.
    Args:
        runs (int): test each bandit with the # of runs.
        time (int): the # of the time-steps in each run.
    """
    epsilons = [0.0, 0.1]
    initials = [5, 0]
    bandits = []
    for e, i in zip(epsilons, initials):
        bandits.append(Bandit(epsilon=e, optimistic_init=i, step_size=0.1))

    print("===== %s =====" % ("Optimistic initial"))
    best_action_counts, _ = simulate(bandits, runs, time)

    for n, (e, i) in enumerate(zip(epsilons, initials)):
        plt.plot(best_action_counts[n],
                 label="epsilon = %.2f, q = %d" % (e, i))
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig(os.path.join(SAVING_PATH, "figure_2_3_optimistic_initial.png"))
    plt.close()


def upper_confidence_bound(runs=2000, time=1000):
    """Test the k-armed bandit with UCB.
    Args:
        runs (int): test each bandit with the # of runs.
        time (int): the # of the time-steps in each run.
    """
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(bandits, runs, time)

    plt.plot(average_rewards[0], label='UCB c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig(os.path.join(SAVING_PATH, "figure_2_4_UCB.png"))
    plt.close()


def gradient(runs=2000, time=1000):
    """Test the k-armed bandit with gradient.
    Args:
        runs (int): test each bandit with the # of runs.
        time (int): the # of the time-steps in each run.
    """
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1,
                          gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1,
                          gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4,
                          gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4,
                          gradient_baseline=False, true_reward=4))

    print("===== %s =====" % ("Gradient"))
    best_action_counts, _ = simulate(bandits, runs, time)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline', ]

    for i in range(0, len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig(os.path.join(SAVING_PATH, "figure_2_5_gradient.png"))
    plt.close()


def compare_all(runs=2000, time=1000):
    """Compare all algorithms.
    """
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, optimistic_init=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(bandits, runs, time)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig(os.path.join(SAVING_PATH, "figure_2_6_compare_all.png"))
    plt.close()


if __name__ == '__main__':
    if not os.path.isdir(SAVING_PATH):
        os.makedirs(SAVING_PATH)

    reward_distribution()
    optimistic_initial()
    upper_confidence_bound()
    gradient()
    compare_all()
