#!/usr/bin/env python3
"""
Test script for the meta-learning robot implementation.
This script tests both the basic MAML robot and the Hebbian-enhanced version.
"""

import sys
import os
import time
import numpy as np
from maze import Maze
from meta_learning_robot import MetaLearningRobot
from hebbian_meta_robot import HebbianMetaLearningRobot

def test_meta_learning_robot(maze_file, robot_class, robot_name):
    """
    Test a meta-learning robot on a given maze
    """
    print(f"\n{'='*60}")
    print(f"Testing {robot_name} on {maze_file}")
    print(f"{'='*60}")
    
    try:
        # Create maze
        test_maze = Maze(maze_file)
        print(f"Maze dimensions: {test_maze.dim}x{test_maze.dim}")
        
        # Initialize robot
        robot = robot_class(test_maze.dim)
        print(f"Robot initialized with {robot_name}")
        
        # Test parameters
        max_time = 1000
        total_time = 0
        runtimes = []
        
        # Global dictionaries for robot movement and sensing
        dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
                      'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
                      'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
                      'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
        dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
                   'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
        
        # Run two phases: training and testing
        for run in range(2):
            print(f"\nStarting run {run + 1} ({'Training' if run == 0 else 'Testing'})")
            
            # Reset robot
            robot_pos = {'location': [0, 0], 'heading': 'up'}
            run_active = True
            hit_goal = False
            run_start_time = time.time()
            
            while run_active:
                # Check for end of time
                total_time += 1
                if total_time > max_time:
                    run_active = False
                    print("Allotted time exceeded.")
                    break
                
                # Provide robot with sensor information, get actions
                sensing = [test_maze.dist_to_wall(robot_pos['location'], heading)
                          for heading in dir_sensors[robot_pos['heading']]]
                
                rotation, movement = robot.next_move(sensing)
                
                # Check for a reset
                if (rotation, movement) == ('Reset', 'Reset'):
                    if run == 0 and hit_goal:
                        run_active = False
                        runtimes.append(total_time)
                        print("Ending first run. Starting next run.")
                        break
                    elif run == 0 and not hit_goal:
                        print("Cannot reset - robot has not hit goal yet.")
                        continue
                    else:
                        print("Cannot reset on runs after the first.")
                        continue
                
                # Perform rotation
                if rotation == -90:
                    robot_pos['heading'] = dir_sensors[robot_pos['heading']][0]
                elif rotation == 90:
                    robot_pos['heading'] = dir_sensors[robot_pos['heading']][2]
                elif rotation == 0:
                    pass
                else:
                    print("Invalid rotation value, no rotation performed.")
                
                # Perform movement
                if abs(movement) > 3:
                    print("Movement limited to three squares in a turn.")
                movement = max(min(int(movement), 3), -3)  # fix to range [-3, 3]
                
                while movement:
                    if movement > 0:
                        if test_maze.is_permissible(robot_pos['location'], robot_pos['heading']):
                            robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
                            robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
                            movement -= 1
                        else:
                            print("Movement stopped by wall.")
                            movement = 0
                    else:
                        rev_heading = dir_sensors[robot_pos['heading']][1]  # Opposite direction
                        if test_maze.is_permissible(robot_pos['location'], rev_heading):
                            robot_pos['location'][0] += dir_move[rev_heading][0]
                            robot_pos['location'][1] += dir_move[rev_heading][1]
                            movement += 1
                        else:
                            print("Movement stopped by wall.")
                            movement = 0
                
                # Check if goal reached
                if (robot_pos['location'][0] == test_maze.dim//2 - 1 and 
                    robot_pos['location'][1] == test_maze.dim//2):
                    if not hit_goal:
                        hit_goal = True
                        run_time = time.time() - run_start_time
                        print(f"Goal reached in run {run + 1}! Time: {run_time:.2f}s")
                        print(f"Location: {robot_pos['location']}")
                
                # Update robot position for next iteration
                robot.x = robot_pos['location'][0]
                robot.y = robot_pos['location'][1]
                robot.heading = robot_pos['heading']
                robot.location = robot_pos['location']
            
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            print(f"Run {run + 1} completed in {run_duration:.2f}s")
            
            # Print meta-learning statistics
            if hasattr(robot, 'get_meta_learning_stats'):
                stats = robot.get_meta_learning_stats()
                print(f"\nMeta-learning statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        # Final results
        print(f"\n{'='*60}")
        print(f"Final Results for {robot_name}")
        print(f"{'='*60}")
        print(f"Total time: {total_time}")
        print(f"Goal reached: {hit_goal}")
        if runtimes:
            print(f"Training run time: {runtimes[0]}")
        
        return {
            'robot_name': robot_name,
            'maze_file': maze_file,
            'total_time': total_time,
            'goal_reached': hit_goal,
            'runtimes': runtimes,
            'success': hit_goal and total_time <= max_time
        }
        
    except Exception as e:
        print(f"Error testing {robot_name}: {str(e)}")
        return {
            'robot_name': robot_name,
            'maze_file': maze_file,
            'error': str(e),
            'success': False
        }

def compare_robots(maze_files):
    """
    Compare the performance of different robot implementations
    """
    print(f"\n{'='*80}")
    print("META-LEARNING ROBOT COMPARISON TEST")
    print(f"{'='*80}")
    
    results = []
    
    # Test each maze with each robot type
    for maze_file in maze_files:
        print(f"\nTesting maze: {maze_file}")
        
        # Test basic MAML robot
        result_maml = test_meta_learning_robot(maze_file, MetaLearningRobot, "MAML Robot")
        results.append(result_maml)
        
        # Test Hebbian-enhanced robot
        result_hebbian = test_meta_learning_robot(maze_file, HebbianMetaLearningRobot, "Hebbian MAML Robot")
        results.append(result_hebbian)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        if 'error' in result:
            print(f"{result['robot_name']} on {result['maze_file']}: ERROR - {result['error']}")
        else:
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"{result['robot_name']} on {result['maze_file']}: {status}")
            print(f"  Total time: {result['total_time']}, Goal reached: {result['goal_reached']}")
    
    # Calculate success rates
    maml_results = [r for r in results if 'MAML Robot' in r['robot_name'] and 'error' not in r]
    hebbian_results = [r for r in results if 'Hebbian MAML Robot' in r['robot_name'] and 'error' not in r]
    
    if maml_results:
        maml_success_rate = sum(1 for r in maml_results if r['success']) / len(maml_results) * 100
        print(f"\nMAML Robot Success Rate: {maml_success_rate:.1f}%")
    
    if hebbian_results:
        hebbian_success_rate = sum(1 for r in hebbian_results if r['success']) / len(hebbian_results) * 100
        print(f"Hebbian MAML Robot Success Rate: {hebbian_success_rate:.1f}%")

def main():
    """
    Main function to run the meta-learning robot tests
    """
    # Check if maze files are provided as arguments
    if len(sys.argv) > 1:
        maze_files = sys.argv[1:]
    else:
        # Use default maze files
        maze_files = ['test_maze_01.txt', 'test_maze_02.txt', 'test_maze_03.txt']
    
    # Verify maze files exist
    existing_maze_files = []
    for maze_file in maze_files:
        if os.path.exists(maze_file):
            existing_maze_files.append(maze_file)
        else:
            print(f"Warning: Maze file {maze_file} not found, skipping...")
    
    if not existing_maze_files:
        print("No valid maze files found. Please provide valid maze files as arguments.")
        print("Example: python test_meta_learning.py test_maze_01.txt test_maze_02.txt")
        return
    
    print(f"Testing meta-learning robots on {len(existing_maze_files)} maze(s): {existing_maze_files}")
    
    # Run comparison test
    compare_robots(existing_maze_files)
    
    print(f"\n{'='*80}")
    print("Meta-learning robot testing completed!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
