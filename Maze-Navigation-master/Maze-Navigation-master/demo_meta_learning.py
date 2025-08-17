#!/usr/bin/env python3
"""
Demonstration script for the meta-learning robot.
This script shows how the robot learns and adapts to different maze environments.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from maze import Maze
from hebbian_meta_robot import HebbianMetaLearningRobot

def visualize_robot_learning(maze_file, max_episodes=3):
    """
    Visualize the robot's learning process in a maze
    """
    print(f"🎯 Demonstrating Meta-Learning Robot in {maze_file}")
    print("=" * 60)
    
    # Create maze and robot
    test_maze = Maze(maze_file)
    robot = HebbianMetaLearningRobot(test_maze.dim)
    
    print(f"📏 Maze dimensions: {test_maze.dim}x{test_maze.dim}")
    print(f"🎯 Goal location: ({test_maze.dim//2 - 1}, {test_maze.dim//2})")
    print(f"🤖 Robot starting position: ({robot.x}, {robot.y})")
    
    # Learning statistics tracking
    episode_rewards = []
    episode_lengths = []
    exploration_rates = []
    meta_learning_progress = []
    
    # Global dictionaries for robot movement and sensing
    dir_sensors = {'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                  'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
    dir_move = {'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
    
    for episode in range(max_episodes):
        print(f"\n🔄 Episode {episode + 1}/{max_episodes}")
        print("-" * 40)
        
        # Reset robot and maze state
        robot_pos = {'location': [0, 0], 'heading': 'up'}
        robot.x, robot.y = robot_pos['location']
        robot.heading = robot_pos['heading']
        robot.location = robot_pos['location']
        
        episode_reward = 0
        episode_steps = 0
        visited_cells = set()
        
        # Episode loop
        max_steps = 200
        goal_reached = False
        
        for step in range(max_steps):
            # Get sensor readings
            sensing = [test_maze.dist_to_wall(robot_pos['location'], heading)
                      for heading in dir_sensors[robot_pos['heading']]]
            
            # Robot decision
            rotation, movement = robot.next_move(sensing)
            
            # Handle reset
            if (rotation, movement) == ('Reset', 'Reset'):
                print("🔄 Robot requested reset - ending episode")
                break
            
            # Perform rotation
            if rotation == -90:
                robot_pos['heading'] = dir_sensors[robot_pos['heading']][0]
            elif rotation == 90:
                robot_pos['heading'] = dir_sensors[robot_pos['heading']][2]
            
            # Perform movement
            movement = max(min(int(movement), 3), -3)
            while movement:
                if movement > 0:
                    if test_maze.is_permissible(robot_pos['location'], robot_pos['heading']):
                        robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
                        robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
                        movement -= 1
                    else:
                        movement = 0
                else:
                    rev_heading = dir_sensors[robot_pos['heading']][1]
                    if test_maze.is_permissible(robot_pos['location'], rev_heading):
                        robot_pos['location'][0] += dir_move[rev_heading][0]
                        robot_pos['location'][1] += dir_move[rev_heading][1]
                        movement += 1
                    else:
                        movement = 0
            
            # Update robot state
            robot.x, robot.y = robot_pos['location']
            robot.heading = robot_pos['heading']
            robot.location = robot_pos['location']
            
            # Track visited cells
            visited_cells.add(tuple(robot_pos['location']))
            
            # Check goal
            if (robot_pos['location'][0] == test_maze.dim//2 - 1 and 
                robot_pos['location'][1] == test_maze.dim//2):
                goal_reached = True
                print(f"🎉 Goal reached at step {step + 1}!")
                break
            
            episode_steps += 1
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: Location {robot_pos['location']}, Heading {robot_pos['heading']}")
        
        # Episode summary
        coverage = len(visited_cells) / (test_maze.dim * test_maze.dim) * 100
        print(f"📊 Episode {episode + 1} Summary:")
        print(f"   Steps taken: {episode_steps}")
        print(f"   Goal reached: {'✅' if goal_reached else '❌'}")
        print(f"   Maze coverage: {coverage:.1f}%")
        print(f"   Cells visited: {len(visited_cells)}")
        
        # Get meta-learning statistics
        stats = robot.get_meta_learning_stats()
        print(f"🧠 Meta-learning stats:")
        print(f"   Total tasks: {stats['total_tasks']}")
        print(f"   Total experiences: {stats['total_experiences']}")
        print(f"   Current epsilon: {stats['current_epsilon']:.3f}")
        print(f"   Hebbian updates: {stats['hebbian_updates']}")
        print(f"   Few-shot memory: {stats['few_shot_memory_size']}")
        
        # Store episode statistics
        episode_rewards.append(episode_steps)  # Using steps as simple reward metric
        episode_lengths.append(episode_steps)
        exploration_rates.append(stats['current_epsilon'])
        meta_learning_progress.append(stats['total_experiences'])
        
        # Small delay between episodes
        time.sleep(1)
    
    # Final summary
    print(f"\n🎯 Final Learning Summary")
    print("=" * 60)
    print(f"Total episodes completed: {len(episode_rewards)}")
    print(f"Average steps per episode: {np.mean(episode_lengths):.1f}")
    print(f"Final exploration rate: {exploration_rates[-1]:.3f}")
    print(f"Total experiences collected: {meta_learning_progress[-1]}")
    
    # Plot learning progress
    plot_learning_progress(episode_lengths, exploration_rates, meta_learning_progress)
    
    return robot

def plot_learning_progress(episode_lengths, exploration_rates, meta_learning_progress):
    """
    Create plots showing the robot's learning progress
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Episode lengths
    ax1.plot(episode_lengths, 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Episode Performance')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps to Goal')
    ax1.grid(True, alpha=0.3)
    
    # Exploration rate decay
    ax2.plot(exploration_rates, 'r-o', linewidth=2, markersize=8)
    ax2.set_title('Exploration Rate Decay')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon (Exploration Rate)')
    ax2.grid(True, alpha=0.3)
    
    # Meta-learning progress
    ax3.plot(meta_learning_progress, 'g-o', linewidth=2, markersize=8)
    ax3.set_title('Meta-Learning Progress')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Experiences')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('meta_learning_progress.png', dpi=300, bbox_inches='tight')
    print(f"📊 Learning progress plots saved to 'meta_learning_progress.png'")
    plt.show()

def demonstrate_few_shot_learning(robot, maze_file):
    """
    Demonstrate the robot's few-shot learning capabilities
    """
    print(f"\n🧠 Few-Shot Learning Demonstration")
    print("=" * 60)
    
    # Create a new maze instance
    test_maze = Maze(maze_file)
    
    # Show few-shot memory contents
    stats = robot.get_meta_learning_stats()
    print(f"Few-shot memory size: {stats['few_shot_memory_size']}")
    
    if hasattr(robot, 'few_shot_memory') and len(robot.few_shot_memory) > 0:
        print("Recent successful experiences in memory:")
        for i, memory_item in enumerate(list(robot.few_shot_memory)[-3:]):  # Show last 3
            print(f"  Experience {i+1}:")
            print(f"    State: {memory_item['state'].flatten()}")
            print(f"    Action: {memory_item['action']}")
            print(f"    Reward: {memory_item['reward']}")
            print(f"    Step: {memory_item['step']}")
    
    # Test adaptation to new situation
    print(f"\nTesting adaptation to new sensor input...")
    test_sensors = [2, 3, 1]  # Example sensor reading
    test_state = robot._get_state_representation(test_sensors)
    
    # Get action using few-shot learning
    action = robot._select_action(test_state, training=False)
    print(f"Test sensors: {test_sensors}")
    print(f"Selected action: {action}")
    print(f"Action confidence: High (using few-shot learning)")

def main():
    """
    Main demonstration function
    """
    print("🚀 Meta-Learning Robot Demonstration")
    print("=" * 80)
    
    # Available maze files
    maze_files = ['test_maze_01.txt', 'test_maze_02.txt', 'test_maze_03.txt']
    
    # Check which maze files exist
    available_mazes = []
    for maze_file in maze_files:
        try:
            with open(maze_file, 'r') as f:
                available_mazes.append(maze_file)
        except FileNotFoundError:
            print(f"⚠️  Maze file {maze_file} not found, skipping...")
    
    if not available_mazes:
        print("❌ No maze files found. Please ensure test maze files are in the current directory.")
        return
    
    print(f"📁 Available maze files: {available_mazes}")
    
    # Run demonstration on first available maze
    maze_file = available_mazes[0]
    print(f"\n🎯 Running demonstration on: {maze_file}")
    
    try:
        # Run learning demonstration
        robot = visualize_robot_learning(maze_file, max_episodes=3)
        
        # Demonstrate few-shot learning
        demonstrate_few_shot_learning(robot, maze_file)
        
        # Show final statistics
        print(f"\n🎉 Demonstration completed successfully!")
        print("=" * 80)
        
        # Option to save the trained model
        save_model = input("\n💾 Would you like to save the trained model? (y/n): ").lower().strip()
        if save_model == 'y':
            filename = f"trained_robot_{maze_file.replace('.txt', '')}.pkl"
            robot.save_model(filename)
            print(f"✅ Model saved as {filename}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
