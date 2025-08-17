import numpy as np
import random
from collections import deque

class StandardQLearningRobot(object):
    """
    A standard Q-learning robot for maze navigation.
    This implementation uses traditional Q-learning without meta-learning capabilities.
    """
    
    def __init__(self, maze_dim):
        '''
        Initialize the standard Q-learning robot
        '''
        # Position and orientation
        self.x = maze_dim - 1
        self.y = 0
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.location = [self.x, self.y]
        self.moves = 0
        self.run = 0
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.005
        
        # State and action spaces
        self.num_states = maze_dim * maze_dim * 4  # position (x,y) + heading
        self.num_actions = 4  # up, right, down, left
        
        # Q-table: state -> action -> Q-value
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # Experience tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.learning_progress = []
        
        # Maze mapping
        self.maze_map = np.zeros((maze_dim, maze_dim))
        self.visited = np.zeros((maze_dim, maze_dim))
        self.goal = [maze_dim//2 - 1, maze_dim//2]
        
        # Performance tracking
        self.total_episodes = 0
        self.successful_episodes = 0
        
    def _get_state_index(self, x, y, heading):
        """Convert position and heading to state index"""
        headings = ['up', 'right', 'down', 'left']
        heading_idx = headings.index(heading)
        return (x * self.maze_dim + y) * 4 + heading_idx
    
    def _get_state_representation(self, sensors):
        """Convert sensor readings and robot state to state index"""
        return self._get_state_index(self.x, self.y, self.heading)
    
    def _select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Exploitation: choose action with highest Q-value
        return np.argmax(self.q_table[state, :])
    
    def _action_to_movement(self, action):
        """Convert neural network action to robot movement"""
        actions = ['up', 'right', 'down', 'left']
        target_direction = actions[action]
        
        # Calculate rotation needed
        current_idx = actions.index(self.heading)
        target_idx = actions.index(target_direction)
        
        rotation = ((target_idx - current_idx) * 90) % 360
        if rotation > 180:
            rotation -= 360
            
        # Movement distance (1-3 squares based on sensor readings)
        movement = 1  # Default to 1, can be enhanced with sensor readings
        
        return rotation, movement
    
    def _compute_reward(self, sensors, action):
        """Compute reward based on current state and action"""
        reward = -0.1  # Small negative reward for each step
        
        # Reward for moving towards unvisited areas
        x, y = self.location
        if self.visited[x][y] == 0:
            reward += 1.0
            
        # Reward for reaching goal
        if x == self.goal[0] and y == self.goal[1]:
            reward += 100.0
            
        # Penalty for hitting walls
        if any(s == 0 for s in sensors):
            reward -= 5.0
            
        return reward
    
    def _update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        
        return td_error
    
    def _update_maze_map(self, sensors):
        """Update internal maze representation based on sensor readings"""
        x, y = self.location
        self.visited[x][y] = 1
        
        # Simple wall detection based on sensor readings
        directions = ['left', 'up', 'right']
        for i, direction in enumerate(directions):
            if sensors[i] == 0:  # Wall detected
                # Mark wall in maze map (simplified)
                pass
    
    def reset(self):
        """Reset robot to start position"""
        self.x = self.maze_dim - 1
        self.y = 0
        self.heading = 'up'
        self.location = [self.x, self.y]
        self.moves = 0
        
        print("Reset! Starting new episode")
        return ('Reset', 'Reset')
    
    def next_move(self, sensors):
        """
        Determine next move using standard Q-learning approach
        """
        # Update maze map
        self._update_maze_map(sensors)
        
        # Get current state representation
        current_state = self._get_state_representation(sensors)
        
        # Select action using current policy
        action_idx = self._select_action(current_state, training=(self.run == 0))
        
        # Convert action to robot movement
        rotation, movement = self._action_to_movement(action_idx)
        
        # Store experience for Q-learning
        if self.run == 0:  # Only during training
            # Simulate next state (simplified)
            next_state = current_state  # In practice, this would be the actual next state
            
            # Compute reward
            reward = self._compute_reward(sensors, action_idx)
            
            # Update Q-table
            td_error = self._update_q_table(current_state, action_idx, reward, next_state)
            
            # Track learning progress
            self.learning_progress.append(abs(td_error))
        
        # Check if goal reached
        if self.x == self.goal[0] and self.y == self.goal[1]:
            print(f"Goal reached!")
            if self.run == 0:
                # End training run
                self.successful_episodes += 1
                if self.moves > 50:  # Continue training for a while
                    self.reset()
                    self.run += 1
                    return ('Reset', 'Reset')
        
        # Update position and heading
        self._update_position(rotation, movement)
        self.moves += 1
        
        # Decay exploration rate
        if self.run == 0:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.total_episodes)
        
        print(f'Rotation: {rotation}, Movement: {movement}, Location: {self.location}, Heading: {self.heading}, Moves: {self.moves}')
        return rotation, movement
    
    def _update_position(self, rotation, movement):
        """Update robot position based on rotation and movement"""
        # Update heading
        headings = ['up', 'right', 'down', 'left']
        current_idx = headings.index(self.heading)
        
        if rotation == 90:
            new_idx = (current_idx + 1) % 4
        elif rotation == -90:
            new_idx = (current_idx - 1) % 4
        else:
            new_idx = current_idx
            
        self.heading = headings[new_idx]
        
        # Update position
        if self.heading == 'up':
            self.x = max(0, self.x - movement)
        elif self.heading == 'right':
            self.y = min(self.maze_dim - 1, self.y + movement)
        elif self.heading == 'down':
            self.x = min(self.maze_dim - 1, self.x + movement)
        elif self.heading == 'left':
            self.y = max(0, self.y - movement)
            
        self.location = [self.x, self.y]
    
    def get_q_learning_stats(self):
        """Return Q-learning performance statistics"""
        return {
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.total_episodes),
            'current_epsilon': self.epsilon,
            'q_table_norm': np.linalg.norm(self.q_table),
            'avg_td_error': np.mean(self.learning_progress) if self.learning_progress else 0.0,
            'q_table_size': self.q_table.shape
        }
    
    def save_q_table(self, filename):
        """Save the trained Q-table"""
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename):
        """Load a previously trained Q-table"""
        self.q_table = np.load(filename)
        print(f"Q-table loaded from {filename}")
    
    def get_optimal_policy(self):
        """Get the optimal policy from the Q-table"""
        return np.argmax(self.q_table, axis=1)
    
    def evaluate_policy(self, num_episodes=100):
        """Evaluate the current policy without exploration"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # No exploration
        
        success_count = 0
        total_steps = 0
        
        for episode in range(num_episodes):
            # Reset environment
            self.x = self.maze_dim - 1
            self.y = 0
            self.heading = 'up'
            self.location = [self.x, self.y]
            
            steps = 0
            max_steps = 200
            
            while steps < max_steps:
                # Get current state
                current_state = self._get_state_representation([1, 1, 1])  # Dummy sensors
                
                # Take optimal action
                action = np.argmax(self.q_table[current_state, :])
                rotation, movement = self._action_to_movement(action)
                
                # Update position
                self._update_position(rotation, movement)
                steps += 1
                
                # Check if goal reached
                if self.x == self.goal[0] and self.y == self.goal[1]:
                    success_count += 1
                    break
                
                # Check if stuck
                if steps > 50 and self.x == self.maze_dim - 1 and self.y == 0:
                    break
            
            total_steps += steps
        
        # Restore exploration
        self.epsilon = original_epsilon
        
        return {
            'success_rate': success_count / num_episodes,
            'avg_steps': total_steps / num_episodes,
            'success_count': success_count,
            'total_episodes': num_episodes
        }

