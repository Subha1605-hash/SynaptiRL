#!/usr/bin/env python3
"""
Comprehensive comparison between Standard Q-Learning and Meta-Learning approaches
for maze navigation tasks.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, using basic matplotlib styling")
from maze import Maze
from standard_q_learning_robot import StandardQLearningRobot
from hebbian_meta_robot import HebbianMetaLearningRobot

class ApproachComparison:
    """
    Compare different learning approaches on maze navigation
    """
    
    def __init__(self, maze_file):
        self.maze_file = maze_file
        self.maze = Maze(maze_file)
        self.results = {}
        
    def test_standard_q_learning(self, num_episodes=100):
        """
        Test standard Q-learning approach
        """
        print(f"\n🧠 Testing Standard Q-Learning on {self.maze_file}")
        print("=" * 60)
        
        robot = StandardQLearningRobot(self.maze.dim)
        
        # Training phase
        training_results = self._train_robot(robot, num_episodes, "Standard Q-Learning")
        
        # Evaluation phase
        evaluation_results = robot.evaluate_policy(num_episodes=50)
        
        # Store results
        self.results['standard_q_learning'] = {
            'training': training_results,
            'evaluation': evaluation_results,
            'robot': robot
        }
        
        return robot
    
    def test_meta_learning(self, num_episodes=100):
        """
        Test meta-learning approach
        """
        print(f"\n🚀 Testing Meta-Learning on {self.maze_file}")
        print("=" * 60)
        
        robot = HebbianMetaLearningRobot(self.maze.dim)
        
        # Training phase
        training_results = self._train_robot(robot, num_episodes, "Meta-Learning")
        
        # Evaluation phase (simulated since meta-learning doesn't have evaluate_policy)
        evaluation_results = self._evaluate_meta_learning(robot, num_episodes=50)
        
        # Store results
        self.results['meta_learning'] = {
            'training': training_results,
            'evaluation': evaluation_results,
            'robot': robot
        }
        
        return robot
    
    def _train_robot(self, robot, num_episodes, approach_name):
        """
        Train a robot for specified number of episodes
        """
        print(f"Training {approach_name} for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        learning_metrics = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
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
                sensing = self._get_sensor_readings(robot_pos['location'], robot_pos['heading'])
                
                # Robot decision
                rotation, movement = robot.next_move(sensing)
                
                # Handle reset
                if (rotation, movement) == ('Reset', 'Reset'):
                    break
                
                # Perform rotation and movement
                robot_pos = self._execute_movement(robot_pos, rotation, movement)
                
                # Update robot state
                robot.x, robot.y = robot_pos['location']
                robot.heading = robot_pos['heading']
                robot.location = robot_pos['location']
                
                # Track visited cells
                visited_cells.add(tuple(robot_pos['location']))
                
                # Check goal
                if (robot_pos['location'][0] == self.maze.dim//2 - 1 and 
                    robot_pos['location'][1] == self.maze.dim//2):
                    goal_reached = True
                    break
                
                episode_steps += 1
            
            # Episode summary
            coverage = len(visited_cells) / (self.maze.dim * self.maze.dim) * 100
            episode_rewards.append(episode_steps)  # Using steps as simple reward metric
            episode_lengths.append(episode_steps)
            success_rates.append(1.0 if goal_reached else 0.0)
            
            # Get learning metrics
            if hasattr(robot, 'get_meta_learning_stats'):
                stats = robot.get_meta_learning_stats()
                learning_metrics.append({
                    'epsilon': stats['current_epsilon'],
                    'experiences': stats['total_experiences'],
                    'hebbian_updates': stats['hebbian_updates']
                })
            elif hasattr(robot, 'get_q_learning_stats'):
                stats = robot.get_q_learning_stats()
                learning_metrics.append({
                    'epsilon': stats['current_epsilon'],
                    'td_error': stats['avg_td_error'],
                    'q_table_norm': stats['q_table_norm']
                })
            
            # Print progress every 20 episodes
            if episode % 20 == 0:
                print(f"  Episode {episode}: Steps {episode_steps}, Goal: {'✅' if goal_reached else '❌'}, Coverage: {coverage:.1f}%")
        
        training_time = time.time() - start_time
        
        # Calculate training statistics
        training_results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rates': success_rates,
            'learning_metrics': learning_metrics,
            'training_time': training_time,
            'avg_success_rate': np.mean(success_rates),
            'avg_episode_length': np.mean(episode_lengths),
            'final_success_rate': np.mean(success_rates[-10:]) if len(success_rates) >= 10 else np.mean(success_rates)
        }
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Final success rate: {training_results['final_success_rate']:.3f}")
        print(f"Average episode length: {training_results['avg_episode_length']:.1f}")
        
        return training_results
    
    def _evaluate_meta_learning(self, robot, num_episodes=50):
        """
        Evaluate meta-learning robot performance
        """
        print(f"Evaluating Meta-Learning robot for {num_episodes} episodes...")
        
        success_count = 0
        total_steps = 0
        
        for episode in range(num_episodes):
            # Reset robot and maze state
            robot_pos = {'location': [0, 0], 'heading': 'up'}
            robot.x, robot.y = robot_pos['location']
            robot.heading = robot_pos['heading']
            robot.location = robot_pos['location']
            
            episode_steps = 0
            max_steps = 200
            goal_reached = False
            
            for step in range(max_steps):
                # Get sensor readings
                sensing = self._get_sensor_readings(robot_pos['location'], robot_pos['heading'])
                
                # Robot decision (no exploration during evaluation)
                rotation, movement = robot.next_move(sensing)
                
                # Handle reset
                if (rotation, movement) == ('Reset', 'Reset'):
                    break
                
                # Perform rotation and movement
                robot_pos = self._execute_movement(robot_pos, rotation, movement)
                
                # Update robot state
                robot.x, robot.y = robot_pos['location']
                robot.heading = robot_pos['heading']
                robot.location = robot_pos['location']
                
                # Check goal
                if (robot_pos['location'][0] == self.maze.dim//2 - 1 and 
                    robot_pos['location'][1] == self.maze.dim//2):
                    goal_reached = True
                    break
                
                episode_steps += 1
            
            if goal_reached:
                success_count += 1
            total_steps += episode_steps
        
        return {
            'success_rate': success_count / num_episodes,
            'avg_steps': total_steps / num_episodes,
            'success_count': success_count,
            'total_episodes': num_episodes
        }
    
    def _get_sensor_readings(self, location, heading):
        """
        Get sensor readings for robot position and heading
        """
        dir_sensors = {'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                      'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        
        sensing = [self.maze.dist_to_wall(location, heading)
                  for heading in dir_sensors[heading]]
        return sensing
    
    def _execute_movement(self, robot_pos, rotation, movement):
        """
        Execute rotation and movement for robot
        """
        dir_sensors = {'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                      'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        dir_move = {'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
        
        # Perform rotation
        if rotation == -90:
            robot_pos['heading'] = dir_sensors[robot_pos['heading']][0]
        elif rotation == 90:
            robot_pos['heading'] = dir_sensors[robot_pos['heading']][2]
        
        # Perform movement
        movement = max(min(int(movement), 3), -3)
        while movement:
            if movement > 0:
                if self.maze.is_permissible(robot_pos['location'], robot_pos['heading']):
                    robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
                    robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
                    movement -= 1
                else:
                    movement = 0
            else:
                rev_heading = dir_sensors[robot_pos['heading']][1]
                if self.maze.is_permissible(robot_pos['location'], rev_heading):
                    robot_pos['location'][0] += dir_move[rev_heading][0]
                    robot_pos['location'][1] += dir_move[rev_heading][1]
                    movement += 1
                else:
                    movement = 0
        
        return robot_pos
    
    def run_comparison(self, num_episodes=100):
        """
        Run comparison between both approaches
        """
        print(f"🔬 COMPREHENSIVE APPROACH COMPARISON")
        print(f"Testing on maze: {self.maze_file}")
        print(f"Episodes per approach: {num_episodes}")
        print("=" * 80)
        
        # Test both approaches
        self.test_standard_q_learning(num_episodes)
        self.test_meta_learning(num_episodes)
        
        # Generate comparison report
        self._generate_comparison_report()
        
        # Create visualization
        self._create_comparison_plots()
        
        return self.results
    
    def _generate_comparison_report(self):
        """
        Generate detailed comparison report
        """
        print(f"\n📊 COMPARISON REPORT")
        print("=" * 80)
        
        q_learning = self.results['standard_q_learning']
        meta_learning = self.results['meta_learning']
        
        print(f"Standard Q-Learning Results:")
        print(f"  Training Time: {q_learning['training']['training_time']:.2f}s")
        print(f"  Final Success Rate: {q_learning['training']['final_success_rate']:.3f}")
        print(f"  Average Episode Length: {q_learning['training']['avg_episode_length']:.1f}")
        print(f"  Evaluation Success Rate: {q_learning['evaluation']['success_rate']:.3f}")
        print(f"  Evaluation Avg Steps: {q_learning['evaluation']['avg_steps']:.1f}")
        
        print(f"\nMeta-Learning Results:")
        print(f"  Training Time: {meta_learning['training']['training_time']:.2f}s")
        print(f"  Final Success Rate: {meta_learning['training']['final_success_rate']:.3f}")
        print(f"  Average Episode Length: {meta_learning['training']['avg_episode_length']:.1f}")
        print(f"  Evaluation Success Rate: {meta_learning['evaluation']['success_rate']:.3f}")
        print(f"  Evaluation Avg Steps: {meta_learning['evaluation']['avg_steps']:.1f}")
        
        # Calculate improvements
        success_rate_improvement = (meta_learning['training']['final_success_rate'] - 
                                  q_learning['training']['final_success_rate']) / q_learning['training']['final_success_rate'] * 100
        
        steps_improvement = (q_learning['training']['avg_episode_length'] - 
                           meta_learning['training']['avg_episode_length']) / q_learning['training']['avg_episode_length'] * 100
        
        print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
        print(f"  Success Rate: {success_rate_improvement:+.1f}%")
        print(f"  Episode Length: {steps_improvement:+.1f}%")
        
        # Determine winner
        if meta_learning['training']['final_success_rate'] > q_learning['training']['final_success_rate']:
            print(f"\n🏆 WINNER: Meta-Learning approach")
        elif q_learning['training']['final_success_rate'] > meta_learning['training']['final_success_rate']:
            print(f"\n🏆 WINNER: Standard Q-Learning approach")
        else:
            print(f"\n🤝 TIE: Both approaches performed similarly")
    
    def _create_comparison_plots(self):
        """
        Create comparison visualization plots
        """
        print(f"\n📊 Creating comparison plots...")
        
        q_learning = self.results['standard_q_learning']
        meta_learning = self.results['meta_learning']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Success Rate Comparison
        episodes = range(len(q_learning['training']['success_rates']))
        ax1.plot(episodes, q_learning['training']['success_rates'], 'b-', label='Standard Q-Learning', alpha=0.7)
        ax1.plot(episodes, meta_learning['training']['success_rates'], 'r-', label='Meta-Learning', alpha=0.7)
        ax1.set_title('Success Rate Over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Length Comparison
        ax2.plot(episodes, q_learning['training']['episode_lengths'], 'b-', label='Standard Q-Learning', alpha=0.7)
        ax2.plot(episodes, meta_learning['training']['episode_lengths'], 'r-', label='Meta-Learning', alpha=0.7)
        ax2.set_title('Episode Length Over Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Metrics Comparison
        if q_learning['training']['learning_metrics'] and meta_learning['training']['learning_metrics']:
            q_epsilons = [m['epsilon'] for m in q_learning['training']['learning_metrics']]
            m_epsilons = [m['epsilon'] for m in meta_learning['training']['learning_metrics']]
            
            ax3.plot(episodes[:len(q_epsilons)], q_epsilons, 'b-', label='Standard Q-Learning', alpha=0.7)
            ax3.plot(episodes[:len(m_epsilons)], m_epsilons, 'r-', label='Meta-Learning', alpha=0.7)
            ax3.set_title('Exploration Rate (Epsilon) Over Episodes')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final Performance Comparison
        approaches = ['Standard Q-Learning', 'Meta-Learning']
        success_rates = [q_learning['training']['final_success_rate'], 
                        meta_learning['training']['final_success_rate']]
        avg_steps = [q_learning['training']['avg_episode_length'], 
                    meta_learning['training']['avg_episode_length']]
        
        x = np.arange(len(approaches))
        width = 0.35
        
        ax4.bar(x - width/2, success_rates, width, label='Success Rate', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + width/2, avg_steps, width, label='Avg Steps', alpha=0.7, color='orange')
        
        ax4.set_title('Final Performance Comparison')
        ax4.set_xlabel('Approach')
        ax4.set_ylabel('Success Rate', color='blue')
        ax4_twin.set_ylabel('Average Steps', color='orange')
        ax4.set_xticks(x)
        ax4.set_xticklabels(approaches)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('approach_comparison.png', dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to 'approach_comparison.png'")
        plt.show()

def main():
    """
    Main function to run the comparison
    """
    print("🔬 Maze Navigation Approach Comparison")
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
    
    # Run comparison on first available maze
    maze_file = available_mazes[0]
    print(f"\n🎯 Running comparison on: {maze_file}")
    
    try:
        # Create comparison object
        comparison = ApproachComparison(maze_file)
        
        # Run comparison
        results = comparison.run_comparison(num_episodes=100)
        
        print(f"\n🎉 Comparison completed successfully!")
        print("=" * 80)
        
        # Option to save results
        save_results = input("\n💾 Would you like to save the results? (y/n): ").lower().strip()
        if save_results == 'y':
            import pickle
            filename = f"comparison_results_{maze_file.replace('.txt', '')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"✅ Results saved as {filename}")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
