import numpy as np
from matplotlib import pyplot as plt
class Maze:
    def __init__(self, maze, start_position, goal_position):
        # Initialize Maze object with the provided maze, start_position, and goal position
        
        self.maze_height = maze.shape[0] # Get the height of the maze (number of rows)
        self.maze_width = maze.shape[1]  # Get the width of the maze (number of columns)
        self.start_position = start_position    # Set the start position in the maze as a tuple (x, y)
        self.goal_position = goal_position      # Set the goal position in the maze as a tuple (x, y)
        self.maze_layout = maze;

    def show_maze(self):
        # Visualize the maze using Matplotlib
        plt.figure(figsize=(6,6))

        # Display the maze as an image in grayscale ('gray' colormap)
        plt.imshow(self.maze_layout, cmap='gray')

        # Add start and goal positions as 'S' and 'G'
        plt.text(self.start_position[1], self.start_position[0], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[1], self.goal_position[0], 'G', ha='center', va='center', color='green', fontsize=20)

        # Remove ticks and labels from the axes
        plt.xticks([]), plt.yticks([])

        # Show the plot
        plt.show()

