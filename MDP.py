
import numpy as np
import matplotlib.pyplot as plt
from Maze import Maze


action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_index = [0, 1, 2, 3]
state_values = np.zeros((6, 6))
policy_probs  = np.full((6, 6, 4), 0.25)

def policy(state):
    i, j = state
    return policy_probs[i, j]

def simulate_step(maze_layout, state, action):
    dx, dy = action
    x, y = state
    nx = x + dx;
    ny = y + dy;
    row_max = maze_layout.shape[0]
    col_max = maze_layout.shape[1]
    if 0 <= nx < row_max and 0 <= ny < col_max and maze_layout[nx, ny] == 1:
        next_state = (nx, ny);
        reward = -1;
    else:
        next_state = state
        reward = -1
    if nx ==5 and ny == 4:
        reward = 0
    
    return next_state, reward

def value_iteration(maze, policy_probs, state_values, gamma=0.99, theta=0.000001):
    rows, cols = maze_layout.shape
    delta = float('inf')
    while delta > theta:
        delta = 0;
        for i in range(rows):
            for j in range(cols):
                if maze_layout[i, j] == 1:
                    old_value = state_values[i, j]
                    new_value = 0;
                    action_probabilities = policy_probs[i, j]

                    for ind, action in enumerate(action_space):
                        
                        prob = action_probabilities[ind];
                        next_state, reward = simulate_step(maze_layout, (i, j), action)
                        
                        nx, ny = next_state
                        new_value +=  prob * (reward + gamma*state_values[nx, ny])
                            
                    state_values[i, j] = new_value    
                    delta = max(delta, abs(old_value - new_value))
    return state_values

def policy_improvement(policy_probs, state_values, gamma=0.99):
    rows, cols = maze_layout.shape
    policy_stable = True
    for i in range(rows):
        for j in range(cols):
            if maze_layout[i, j] == 1:
                old_action = policy_probs[i, j].argmax()

                new_action = None
                max_qsa = float("-inf")

                for ind, action in enumerate(action_space):
                    next_state, reward = simulate_step(maze_layout, (i, j), action)
                    nx, ny = next_state
                    qsa = reward + gamma * state_values[nx, ny]
                    if qsa > max_qsa:
                        max_qsa = qsa
                        new_action = ind

                action_probs = np.zeros(4)
                action_probs[new_action] = 1.
                policy_probs[i, j] = action_probs

                if new_action != old_action:
                    policy_stable = False

    return policy_stable


maze_layout = np.array([[1, 1, 1, 1, 1, 0],
                     [1, 0, 1, 0, 1, 1], 
                     [1, 1, 0, 0, 0, 1],
                     [0, 1, 0, 1, 1, 1],
                     [0, 1, 1, 1, 0, 1],
                     [0, 0, 0, 0, 1, 1]])

#maze_layout[maze_layout == 1] = 2;
#maze_layout[maze_layout == 0] = 1;
#maze_layout[maze_layout == 2] = 0;


# Create an instance of the maze and set the starting and ending positions
maze = Maze(maze_layout, (0, 0), (5, 4))
# Visualize the maze
values = value_iteration(maze_layout, policy_probs, state_values)

#policy_stable_check = policy_improvement(policy_probs, state_values)

def policy_iteration(policy_probs, state_values, theta=1e-6, gamma=0.99):
    policy_stable = False

    while not policy_stable:
        values = value_iteration(maze_layout, policy_probs, state_values, gamma, theta)

        policy_stable = policy_improvement(policy_probs, state_values, gamma)

policy_iteration(policy_probs, state_values)
solved_map = np.zeros((6,6))

for i in range(6):
    for j in range(6):
        if policy_probs[i, j, 0] == 1:
            solved_map[i, j] = 0
        if policy_probs[i, j, 1] == 1:
            solved_map[i, j] = 1
        if policy_probs[i, j, 2] == 1:
            solved_map[i, j] = 2
        if policy_probs[i, j, 3] == 1:
            solved_map[i, j] = 3
print(solved_map)
print(np.round(values, 6))

plt.figure(figsize=(5,5))
plt.imshow(maze_layout, cmap='gray')
plt.text(0, 0, 'S', ha='center', va='center', color='green', fontsize=20)
plt.text(4, 5, 'G', ha='center', va='center', color='green', fontsize=20)

for i in range(6):
    for j in range(6):
        if maze_layout[i, j] == 1:
            if solved_map[i, j] == 0:
                plt.text(j, i, 'U', ha='center', va='center', color='red', fontsize=20)
            elif solved_map[i, j] == 1:
                plt.text(j, i, 'D', ha='center', va='center', color='red', fontsize=20)
            elif solved_map[i, j] == 2:
                plt.text(j, i, 'L', ha='center', va='center', color='red', fontsize=20)
            elif solved_map[i, j] == 3:
                plt.text(j, i, 'R', ha='center', va='center', color='red', fontsize=20)
        
plt.xticks([]), plt.yticks([])
plt.show()
            
