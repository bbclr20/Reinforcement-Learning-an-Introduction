import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import os

SAVING_PATH = "../../images/chapter03"

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def step(state, action):
    """Update the current state with the action.
    Args:
        state ([int, int]): x, y position.
        action ([int, int]): displacement.
    """
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def draw(value):
    """Create a table to visualize the state-value function.
    Args:
        value: np array of the state-value function.
    """
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = value.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # add cells
    for (i, j), val in np.ndenumerate(value):
        color = 'white'
        tb.add_cell(i, j, width, height,
                    text=val, loc='center', facecolor=color)

    # add row label
    for i, label in enumerate(range(len(value))):
        tb.add_cell(i, -1, width, height,
                    text=label+1, loc='right', edgecolor='none')

    # add column label
    for j, label in enumerate(range(len(value))):
        tb.add_cell(-1, j, width, height/2,
                    text=label+1, loc='center', edgecolor='none')

    ax.add_table(tb)


def random_policy_evaluation(theta=1e-4):
    """Iterate the state-value function with random strategy until it reaches steady state.
    Args:
        theta (float): a small value act as the iterative constraint.
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros(value.shape)
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * \
                        (reward + DISCOUNT * value[next_i, next_j])
    
        if np.sum(np.abs(value - new_value)) < theta:
            draw(np.round(new_value, decimals=2))
            plt.savefig(os.path.join(
                SAVING_PATH, "figure_3_5_policy_evaluation.png"))
            plt.close()
            break
        value = new_value


def greedy_policy_evaluation(theta=1e-4):
    """.Iterate the state-value function with greedy strategy to find the optimal solution.
    Args:
        theta (float): a small value act as the iterative constraint.
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros(value.shape)
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)

        if np.sum(np.abs(new_value - value)) < theta:
            draw(np.round(new_value, decimals=2))
            plt.savefig(os.path.join(
                SAVING_PATH, "figure_3_8_optimal_value_state.png"))
            plt.close()
            break
        value = new_value


if __name__ == '__main__':
    if not os.path.isdir(SAVING_PATH):
        os.makedirs(SAVING_PATH)

    random_policy_evaluation()
    greedy_policy_evaluation()
