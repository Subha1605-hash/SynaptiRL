# Meta-Learning Robot for Maze Navigation

This project implements advanced meta-learning agents for maze navigation, building upon the existing rule-based robot implementation. The meta-learning approach enables robots to quickly adapt to new maze environments with minimal experience.

## 🧠 Meta-Learning Approaches Implemented

### 1. **MAML (Model-Agnostic Meta-Learning) Robot**
- **File**: `meta_learning_robot.py`
- **Core Concept**: Learns to learn quickly across different maze environments
- **Key Features**:
  - Neural network-based Q-learning with meta-optimization
  - Experience replay buffer for efficient learning
  - Task-specific adaptation with few-shot learning capabilities
  - Automatic exploration vs. exploitation balance

### 2. **Hebbian Meta-Learning Robot**
- **File**: `hebbian_meta_robot.py`
- **Core Concept**: Combines MAML with Hebbian learning for rapid adaptation
- **Key Features**:
  - All MAML capabilities plus Hebbian weight updates
  - Few-shot learning memory for similar situations
  - Enhanced neural plasticity for faster learning
  - Model saving/loading for transfer learning

## 🏗️ Architecture Overview

### Neural Network Structure
```
Input Layer (7 neurons)
├── 3 sensor readings (normalized)
├── 4 heading directions (one-hot encoded)
└── Hidden Layer (16 neurons) with tanh activation
    └── Output Layer (4 neurons) for Q-values
```

### Meta-Learning Components
1. **Inner Loop**: Task-specific adaptation (3 gradient steps)
2. **Outer Loop**: Meta-optimization across multiple tasks
3. **Experience Buffer**: Stores transitions for batch learning
4. **Task Experiences**: Separate experience storage per maze instance

### Hebbian Learning Enhancement
- **Hebbian Rule**: Δw = η × input × output
- **Weight Decay**: Prevents excessive weight growth
- **Thresholding**: Limits update magnitude for stability

## 🚀 Key Features

### Fast Adaptation
- **Few-shot Learning**: Learns from 5-10 examples in new environments
- **Meta-weights**: Pre-optimized for rapid task adaptation
- **Experience Transfer**: Leverages knowledge from previous mazes

### Intelligent Exploration
- **Epsilon-greedy Policy**: Balances exploration vs. exploitation
- **Adaptive Epsilon**: Decays exploration rate over time
- **Reward Shaping**: Encourages exploration of unvisited areas

### Robust Learning
- **Experience Replay**: Breaks temporal correlations
- **Batch Learning**: Stable gradient updates
- **Error Handling**: Graceful degradation under uncertainty

## 📊 Performance Metrics

The robots track several key performance indicators:

```python
stats = robot.get_meta_learning_stats()
# Returns:
{
    'total_tasks': number_of_mazes_solved,
    'total_experiences': total_transitions_stored,
    'current_epsilon': exploration_rate,
    'meta_weights_norm': weight_magnitude,
    'hebbian_updates': number_of_hebbian_updates,
    'few_shot_memory_size': stored_successful_experiences,
    'hebbian_weights_norm': hebbian_weight_magnitude
}
```

## 🧪 Testing and Usage

### Basic Testing
```bash
# Test on default mazes
python test_meta_learning.py

# Test on specific maze
python test_meta_learning.py test_maze_01.txt

# Test on multiple mazes
python test_meta_learning.py test_maze_01.txt test_maze_02.txt test_maze_03.txt
```

### Integration with Existing System
```python
# Replace the original robot with meta-learning version
from hebbian_meta_robot import HebbianMetaLearningRobot

# In tester.py, change:
# testrobot = Robot(testmaze.dim)
# To:
testrobot = HebbianMetaLearningRobot(testmaze.dim)
```

### Model Persistence
```python
# Save trained model
robot.save_model('trained_maze_navigator.pkl')

# Load pre-trained model
robot.load_model('trained_maze_navigator.pkl')
```

## 🔧 Configuration Parameters

### Meta-Learning Parameters
```python
self.alpha = 0.01          # Inner loop learning rate
self.beta = 0.001          # Outer loop learning rate
self.meta_batch_size = 5   # Tasks per meta-update
self.inner_steps = 3       # Gradient steps per task
```

### Hebbian Learning Parameters
```python
self.hebbian_lr = 0.01     # Hebbian learning rate
self.hebbian_decay = 0.99  # Weight decay factor
self.hebbian_threshold = 0.1  # Update magnitude limit
```

### Exploration Parameters
```python
self.epsilon = 0.3          # Initial exploration rate
self.epsilon_decay = 0.995  # Decay factor
self.min_epsilon = 0.01     # Minimum exploration rate
```

## 📈 Learning Process

### Phase 1: Training Run
1. **Exploration**: Robot explores maze with high exploration rate
2. **Experience Collection**: Stores state-action-reward transitions
3. **Meta-updates**: Periodically updates meta-weights
4. **Hebbian Updates**: Rapid weight adaptation during exploration
5. **Goal Achievement**: Continues until goal is reached

### Phase 2: Testing Run
1. **Knowledge Application**: Uses learned meta-weights
2. **Few-shot Decisions**: Leverages similar past experiences
3. **Optimal Navigation**: Follows learned optimal paths
4. **Performance Evaluation**: Measures adaptation success

## 🎯 Advantages Over Rule-Based Approach

| Aspect | Rule-Based Robot | Meta-Learning Robot |
|--------|------------------|---------------------|
| **Adaptation Speed** | Slow (requires complete exploration) | Fast (few-shot learning) |
| **Generalization** | Maze-specific rules | Cross-maze knowledge transfer |
| **Learning** | None (static rules) | Continuous improvement |
| **Robustness** | Brittle to maze variations | Adapts to new environments |
| **Efficiency** | Suboptimal paths | Learns optimal strategies |

## 🔬 Technical Implementation Details

### State Representation
- **Sensor Normalization**: Divides by maze dimension for scale-invariance
- **Heading Encoding**: One-hot encoding for discrete direction representation
- **Feature Concatenation**: Combines sensor and orientation information

### Action Selection
- **Q-value Computation**: Neural network forward pass
- **Epsilon-greedy**: Balances exploration and exploitation
- **Few-shot Fallback**: Uses memory for similar situations

### Learning Algorithm
- **TD(0) Learning**: Temporal difference learning with bootstrapping
- **Experience Replay**: Breaks temporal correlations in data
- **Batch Processing**: Stable gradient computation

## 🚧 Limitations and Future Improvements

### Current Limitations
1. **Simplified Gradient Computation**: Uses finite differences instead of automatic differentiation
2. **Fixed Architecture**: Neural network size is hardcoded
3. **Limited Sensor Processing**: Basic sensor normalization
4. **Single Goal**: Only supports center goal location

### Potential Improvements
1. **PyTorch/TensorFlow Integration**: Automatic differentiation and GPU acceleration
2. **Attention Mechanisms**: Better handling of maze structure
3. **Multi-objective Learning**: Multiple goals and constraints
4. **Hierarchical Learning**: High-level strategy + low-level control
5. **Uncertainty Quantification**: Confidence measures for decisions

## 📚 References and Further Reading

- **MAML Paper**: Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.
- **Hebbian Learning**: Hebb, D. O. (1949). The organization of behavior.
- **Meta-Reinforcement Learning**: Duan, Y., et al. (2016). RL²: Fast reinforcement learning via slow reinforcement learning.

## 🤝 Contributing

To extend the meta-learning implementation:

1. **Add New Meta-Learning Algorithms**: Implement Prototypical Networks, Reptile, etc.
2. **Enhance Neural Architecture**: Add attention, memory, or hierarchical components
3. **Improve Exploration Strategies**: Implement curiosity-driven exploration
4. **Add Multi-robot Scenarios**: Collaborative learning and knowledge sharing

## 📄 License

This implementation extends the original Udacity Maze Navigation project with meta-learning capabilities. The original project structure and maze simulation remain unchanged.
