import numpy as np
import matplotlib.pyplot as plt
from Maze import Maze


class MC_ONP:
    
    def __init__(self, maze, start, end, gamma = 0.99, epsilon = 0.2, episodes = 2000):
        self.maze = maze;
        self.row_max = maze.shape[0]
        self.col_max = maze.shape[1]
        self.gamma = gamma;
        self.epsilon = epsilon
        self.episodes = episodes;
        self.Qsa = np.zeros((6, 6, 4));
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.start = start;
        self.end = end;
    
    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            i, j = state
            av = self.Qsa[i, j, :]
            return np.random.choice(np.flatnonzero(av == av.max()))
    
    def step(self, state, action_ind):
        ex, ey = self.end
        action = self.action_space[action_ind]
        x, y = state
        dx, dy = action
        nx = x + dx;
        ny = y + dy;
        row_max = self.maze.shape[0]
        col_max = self.maze.shape[1]
        done = False
        if 0 <= nx < row_max and 0 <= ny < col_max and self.maze[nx, ny] == 1:
            next_state = (nx, ny);
            reward = -1;
        else:
            next_state = state
            reward = -1
        if nx == ex and ny == ey:
            reward = 0
            done = True
        
        return next_state, reward, done
    
    def Run_MC(self):
        sa_returns = {}
        
        for episode in range(1, self.episodes + 1):
            state = self.start
            done = False;
            transitions = [];
            
            while not done:
                action = self.policy(state);
                next_state, reward, done = self.step(state, action)
                transitions.append([state, action, reward])
                state = next_state
                
            G = 0
            for state_t, action_t, reward_t in reversed(transitions):
                G = reward_t + self.gamma*G
                
                if not (state_t, action_t) in sa_returns:
                    sa_returns[(state_t, action_t)] = []
                sa_returns[(state_t, action_t)].append(G)
                x, y = state_t
                self.Qsa[x, y, action_t] = np.mean(sa_returns[(state_t, action_t)])
            print((episode, self.episodes))
                
    
maze_layout = np.array([[1, 1, 1, 1, 1, 0],
                     [1, 0, 1, 0, 1, 1], 
                     [1, 1, 0, 0, 0, 1],
                     [0, 1, 0, 1, 0, 1],
                     [0, 1, 1, 1, 0, 1],
                     [0, 0, 0, 0, 1, 1]])

start = (0, 0);
end = (5, 4);
Monte_Carlo = MC_ONP(maze_layout, start, end)
Monte_Carlo.Run_MC()
solved_map = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        solved_map[i, j] = np.argmax(Monte_Carlo.Qsa[i, j, :])
        
        

plt.figure(figsize=(6,6))
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
            

print(solved_map)