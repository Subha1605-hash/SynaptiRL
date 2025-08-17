import numpy as np
import random
import copy
from collections import deque
import math

class MetaLearningRobot(object):
    """
    A meta-learning robot that uses MAML (Model-Agnostic Meta-Learning) to quickly
    adapt to new maze environments with minimal experience.
    """
    
    def __init__(self, maze_dim):
        '''
        Initialize the meta-learning robot with MAML capabilities
        '''
        # Position and orientation
        self.x = maze_dim - 1
        self.y = 0
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.location = [self.x, self.y]
        self.moves = 0
        self.run = 0
        
        # Meta-learning parameters
        self.alpha = 0.01  # Inner loop learning rate
        self.beta = 0.001  # Outer loop learning rate
        self.meta_batch_size = 5  # Number of tasks for meta-update
        self.inner_steps = 3  # Number of gradient steps per task
        
        # Neural network parameters (simplified for maze navigation)
        self.input_size = 7  # 3 sensors + 4 heading directions
        self.hidden_size = 16
        self.output_size = 4  # 4 possible actions
        
        # Initialize neural network weights
        self.weights = self._initialize_weights()
        self.meta_weights = copy.deepcopy(self.weights)
        
        # Experience replay for meta-learning
        self.experience_buffer = deque(maxlen=1000)
        self.task_experiences = {}
        
        # Current task identifier
        self.current_task = None
        self.task_step = 0
        
        # Exploration parameters
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Maze mapping
        self.maze_map = np.zeros((maze_dim, maze_dim))
        self.visited = np.zeros((maze_dim, maze_dim))
        self.goal = [maze_dim//2 - 1, maze_dim//2]
        
        # Performance tracking
        self.episode_rewards = []
        self.meta_losses = []
        
    def _initialize_weights(self):
        """Initialize neural network weights with Xavier initialization"""
        weights = {}
        
        # Input to hidden layer
        weights['W1'] = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        weights['b1'] = np.zeros((1, self.hidden_size))
        
        # Hidden to output layer
        weights['W2'] = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        weights['b2'] = np.zeros((1, self.output_size))
        
        return weights
    
    def _forward_pass(self, state, weights=None):
        """Forward pass through the neural network"""
        if weights is None:
            weights = self.weights
            
        # Input layer
        hidden = np.tanh(np.dot(state, weights['W1']) + weights['b1'])
        
        # Output layer
        output = np.dot(hidden, weights['W2']) + weights['b2']
        
        return output, hidden
    
    def _get_state_representation(self, sensors):
        """Convert sensor readings and robot state to neural network input"""
        # Normalize sensor readings
        sensor_input = np.array(sensors) / self.maze_dim
        
        # One-hot encode heading direction
        heading_encoding = np.zeros(4)
        headings = ['up', 'right', 'down', 'left']
        heading_idx = headings.index(self.heading)
        heading_encoding[heading_idx] = 1.0
        
        # Combine sensor and heading information
        state = np.concatenate([sensor_input, heading_encoding])
        return state.reshape(1, -1)
    
    def _select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Get Q-values from neural network
        q_values, _ = self._forward_pass(state)
        return np.argmax(q_values)
    
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
    
    def _compute_loss(self, states, actions, rewards, next_states, dones):
        """Compute TD(0) loss for meta-learning"""
        current_q_values, _ = self._forward_pass(states)
        next_q_values, _ = self._forward_pass(next_states)
        
        target_q = rewards + (1 - dones) * 0.99 * np.max(next_q_values, axis=1, keepdims=True)
        current_q = np.sum(current_q_values * actions, axis=1, keepdims=True)
        
        loss = np.mean((target_q - current_q) ** 2)
        return loss
    
    def _compute_gradients(self, states, actions, rewards, next_states, dones):
        """Compute gradients for meta-learning update"""
        # Simplified gradient computation using finite differences
        epsilon = 1e-6
        gradients = {}
        
        for key in self.weights.keys():
            gradients[key] = np.zeros_like(self.weights[key])
            
            for i in range(self.weights[key].shape[0]):
                for j in range(self.weights[key].shape[1]):
                    # Perturb weight
                    self.weights[key][i, j] += epsilon
                    loss_plus = self._compute_loss(states, actions, rewards, next_states, dones)
                    
                    self.weights[key][i, j] -= 2 * epsilon
                    loss_minus = self._compute_loss(states, actions, rewards, next_states, dones)
                    
                    # Reset weight
                    self.weights[key][i, j] += epsilon
                    
                    # Compute gradient
                    gradients[key][i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def _meta_update(self, task_experiences):
        """Perform meta-learning update using MAML"""
        if len(task_experiences) < self.meta_batch_size:
            return
            
        # Sample random tasks for meta-update
        sampled_tasks = random.sample(list(task_experiences.keys()), 
                                    min(self.meta_batch_size, len(task_experiences)))
        
        meta_gradients = {}
        for key in self.weights.keys():
            meta_gradients[key] = np.zeros_like(self.weights[key])
        
        # For each task, perform inner loop adaptation and compute meta-gradient
        for task_id in sampled_tasks:
            task_data = task_experiences[task_id]
            if len(task_data) < 10:  # Need minimum experience
                continue
                
            # Create task-specific weights
            task_weights = copy.deepcopy(self.meta_weights)
            
            # Inner loop adaptation
            for _ in range(self.inner_steps):
                # Sample batch from task experience
                batch_size = min(32, len(task_data))
                batch_indices = random.sample(range(len(task_data)), batch_size)
                
                batch_states = np.vstack([task_data[i]['state'] for i in batch_indices])
                batch_actions = np.vstack([task_data[i]['action'] for i in batch_indices])
                batch_rewards = np.vstack([task_data[i]['reward'] for i in batch_indices])
                batch_next_states = np.vstack([task_data[i]['next_state'] for i in batch_indices])
                batch_dones = np.vstack([task_data[i]['done'] for i in batch_indices])
                
                # Compute gradients
                gradients = self._compute_gradients(batch_states, batch_actions, 
                                                batch_rewards, batch_next_states, batch_dones)
                
                # Update task weights
                for key in task_weights.keys():
                    task_weights[key] -= self.alpha * gradients[key]
            
            # Compute meta-gradient (simplified)
            # In practice, this would use the validation loss on the adapted weights
            for key in meta_gradients.keys():
                meta_gradients[key] += (task_weights[key] - self.meta_weights[key]) / len(sampled_tasks)
        
        # Update meta-weights
        for key in self.meta_weights.keys():
            self.meta_weights[key] += self.beta * meta_gradients[key]
        
        # Update current weights to meta-weights
        self.weights = copy.deepcopy(self.meta_weights)
    
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
    
    def reset(self):
        """Reset robot to start position"""
        self.x = self.maze_dim - 1
        self.y = 0
        self.heading = 'up'
        self.location = [self.x, self.y]
        self.moves = 0
        
        # Generate new task ID for meta-learning
        self.current_task = f"maze_{self.maze_dim}_{random.randint(1000, 9999)}"
        self.task_step = 0
        
        # Initialize task experience buffer
        if self.current_task not in self.task_experiences:
            self.task_experiences[self.current_task] = []
        
        print(f"Reset! New task: {self.current_task}")
        return ('Reset', 'Reset')
    
    def next_move(self, sensors):
        """
        Determine next move using meta-learning approach
        """
        # Update maze map
        self._update_maze_map(sensors)
        
        # Get current state representation
        current_state = self._get_state_representation(sensors)
        
        # Select action using current policy
        action_idx = self._select_action(current_state, training=(self.run == 0))
        
        # Convert action to robot movement
        rotation, movement = self._action_to_movement(action_idx)
        
        # Store experience for meta-learning
        if self.run == 0:  # Only during training
            # Simulate next state (simplified)
            next_state = current_state.copy()
            
            # Create one-hot action encoding
            action_encoding = np.zeros((1, self.output_size))
            action_encoding[0, action_idx] = 1.0
            
            # Compute reward
            reward = self._compute_reward(sensors, action_idx)
            
            # Store experience
            experience = {
                'state': current_state,
                'action': action_encoding,
                'reward': np.array([[reward]]),
                'next_state': next_state,
                'done': np.array([[0.0]])  # Not done unless goal reached
            }
            
            self.task_experiences[self.current_task].append(experience)
            self.experience_buffer.append(experience)
            
            # Perform meta-update periodically
            if len(self.experience_buffer) % 100 == 0:
                self._meta_update(self.task_experiences)
        
        # Check if goal reached
        if self.x == self.goal[0] and self.y == self.goal[1]:
            print(f"Goal reached! Task: {self.current_task}")
            if self.run == 0:
                # End training run
                if len(self.task_experiences[self.current_task]) > 50:
                    self.reset()
                    self.run += 1
                    return ('Reset', 'Reset')
        
        # Update position and heading
        self._update_position(rotation, movement)
        self.moves += 1
        
        # Decay exploration rate
        if self.run == 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
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
    
    def get_meta_learning_stats(self):
        """Return meta-learning performance statistics"""
        return {
            'total_tasks': len(self.task_experiences),
            'total_experiences': len(self.experience_buffer),
            'current_epsilon': self.epsilon,
            'meta_weights_norm': np.linalg.norm(self.meta_weights['W1'])
        }
