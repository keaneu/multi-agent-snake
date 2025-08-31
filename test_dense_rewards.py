#!/usr/bin/env python3
"""
🎯 Dense Reward System Testing
Alternative approach with immediate feedback and heavy survival incentives
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class DenseRewardConfig:
    """Dense reward configuration for immediate learning feedback"""
    
    # === SURVIVAL REWARDS (Heavy emphasis) ===
    survival_reward_per_step: float = 2.0        # Reward just for staying alive
    wall_avoidance_reward: float = 5.0           # Reward for not hitting walls
    self_collision_avoidance: float = 5.0        # Reward for not hitting self
    
    # === MOVEMENT REWARDS (Immediate feedback) ===
    forward_movement_reward: float = 1.0         # Reward for forward progress
    exploration_reward: float = 0.5              # Reward for visiting new areas
    efficient_movement_reward: float = 3.0       # Reward for direct movement toward food
    
    # === FOOD ACQUISITION (Progressive bonuses) ===
    food_proximity_reward_base: float = 10.0     # Base reward for getting closer to food
    food_proximity_multiplier: float = 2.0       # Multiplier based on distance reduction
    food_consumption_reward: float = 50.0        # Large reward for actually eating food
    
    # === DEATH PENALTIES (Moderate but recoverable) ===
    death_penalty: float = -30.0                 # Death penalty (reduced from -50)
    wall_collision_penalty: float = -15.0        # Specific wall collision penalty
    self_collision_penalty: float = -10.0        # Self collision penalty
    
    # === PROGRESSIVE BONUSES ===
    length_bonus_per_segment: float = 5.0        # Bonus for each body segment
    streak_bonus_multiplier: float = 1.5         # Multiplier for consecutive good actions
    time_survival_bonus: float = 0.1             # Bonus for surviving longer

class DenseRewardCalculator:
    """Calculate dense rewards for immediate learning feedback"""
    
    def __init__(self, config: DenseRewardConfig = None):
        self.config = config or DenseRewardConfig()
        self.previous_positions = {}
        self.food_distances = {}
        self.survival_streaks = {}
        self.visited_positions = {}
        
    def calculate_dense_reward(self, snake_id: str, action_type: str, 
                              game_state: Dict, previous_state: Dict = None) -> float:
        """Calculate immediate dense reward for any action"""
        
        total_reward = 0.0
        reward_breakdown = {}
        
        # Get current snake state
        snake_pos = game_state.get('snake_position', [])
        snake_alive = game_state.get('alive', True)
        foods = game_state.get('foods', [])
        grid_size = game_state.get('grid_size', (20, 20))
        
        if not snake_pos:
            return total_reward
            
        head_pos = tuple(snake_pos[0]) if snake_pos else (0, 0)
        
        # === 1. SURVIVAL REWARDS ===
        if snake_alive:
            # Base survival reward
            survival_reward = self.config.survival_reward_per_step
            total_reward += survival_reward
            reward_breakdown['survival'] = survival_reward
            
            # Time-based survival bonus
            survival_steps = self.survival_streaks.get(snake_id, 0) + 1
            self.survival_streaks[snake_id] = survival_steps
            time_bonus = min(survival_steps * self.config.time_survival_bonus, 10.0)  # Cap at 10
            total_reward += time_bonus
            reward_breakdown['time_survival'] = time_bonus
            
            # Wall avoidance reward
            if self._is_near_wall(head_pos, grid_size):
                if self._is_moving_away_from_wall(head_pos, previous_state, grid_size):
                    wall_avoid = self.config.wall_avoidance_reward
                    total_reward += wall_avoid
                    reward_breakdown['wall_avoidance'] = wall_avoid
            
        else:
            # Death penalties
            self.survival_streaks[snake_id] = 0
            if action_type == "collision_wall":
                penalty = self.config.death_penalty + self.config.wall_collision_penalty
                total_reward += penalty
                reward_breakdown['death_wall'] = penalty
            elif action_type == "collision_self":
                penalty = self.config.death_penalty + self.config.self_collision_penalty
                total_reward += penalty
                reward_breakdown['death_self'] = penalty
            else:
                total_reward += self.config.death_penalty
                reward_breakdown['death'] = self.config.death_penalty
        
        # === 2. MOVEMENT REWARDS ===
        if snake_alive and previous_state:
            prev_head = previous_state.get('snake_position', [[0, 0]])[0]
            prev_head = tuple(prev_head)
            
            # Forward movement reward
            if head_pos != prev_head:
                movement_reward = self.config.forward_movement_reward
                total_reward += movement_reward
                reward_breakdown['movement'] = movement_reward
            
            # Exploration reward
            if snake_id not in self.visited_positions:
                self.visited_positions[snake_id] = set()
            
            if head_pos not in self.visited_positions[snake_id]:
                exploration_reward = self.config.exploration_reward
                total_reward += exploration_reward
                reward_breakdown['exploration'] = exploration_reward
                self.visited_positions[snake_id].add(head_pos)
        
        # === 3. FOOD ACQUISITION REWARDS ===
        if foods and snake_alive:
            # Find nearest food
            nearest_food, min_distance = self._find_nearest_food(head_pos, foods)
            
            if nearest_food:
                # Food proximity reward
                prev_distance = self.food_distances.get(snake_id, min_distance)
                
                if min_distance < prev_distance:
                    # Moving closer to food
                    distance_improvement = prev_distance - min_distance
                    proximity_reward = (self.config.food_proximity_reward_base * 
                                      distance_improvement * self.config.food_proximity_multiplier)
                    total_reward += proximity_reward
                    reward_breakdown['food_proximity'] = proximity_reward
                
                # Efficient movement reward (straight line toward food)
                if self._is_efficient_movement(head_pos, nearest_food, previous_state):
                    efficient_reward = self.config.efficient_movement_reward
                    total_reward += efficient_reward
                    reward_breakdown['efficient_movement'] = efficient_reward
                
                self.food_distances[snake_id] = min_distance
        
        # === 4. FOOD CONSUMPTION REWARD ===
        if action_type == "food_eaten":
            consumption_reward = self.config.food_consumption_reward
            total_reward += consumption_reward
            reward_breakdown['food_consumed'] = consumption_reward
            
            # Reset food distance tracking
            self.food_distances[snake_id] = float('inf')
            
            # Length bonus
            snake_length = len(snake_pos)
            length_bonus = snake_length * self.config.length_bonus_per_segment
            total_reward += length_bonus
            reward_breakdown['length_bonus'] = length_bonus
        
        # === 5. STREAK BONUSES ===
        if len(reward_breakdown) >= 3:  # Multiple positive rewards
            streak_bonus = total_reward * (self.config.streak_bonus_multiplier - 1.0)
            total_reward += streak_bonus
            reward_breakdown['streak_bonus'] = streak_bonus
        
        return total_reward
    
    def _is_near_wall(self, pos: Tuple[int, int], grid_size: Tuple[int, int]) -> bool:
        """Check if position is near a wall"""
        x, y = pos
        width, height = grid_size
        return x <= 1 or x >= width-2 or y <= 1 or y >= height-2
    
    def _is_moving_away_from_wall(self, current_pos: Tuple[int, int], 
                                 previous_state: Dict, grid_size: Tuple[int, int]) -> bool:
        """Check if moving away from wall"""
        if not previous_state:
            return False
            
        prev_pos = previous_state.get('snake_position', [[0, 0]])[0]
        prev_pos = tuple(prev_pos)
        
        # Calculate distances to nearest walls
        current_wall_dist = min(current_pos[0], current_pos[1], 
                               grid_size[0] - current_pos[0] - 1, 
                               grid_size[1] - current_pos[1] - 1)
        prev_wall_dist = min(prev_pos[0], prev_pos[1], 
                            grid_size[0] - prev_pos[0] - 1, 
                            grid_size[1] - prev_pos[1] - 1)
        
        return current_wall_dist > prev_wall_dist
    
    def _find_nearest_food(self, head_pos: Tuple[int, int], 
                          foods: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], float]:
        """Find nearest food and its distance"""
        if not foods:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_food = None
        
        for food in foods:
            food_pos = tuple(food) if isinstance(food, list) else food
            distance = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food_pos
        
        return nearest_food, min_distance
    
    def _is_efficient_movement(self, current_pos: Tuple[int, int], 
                              food_pos: Tuple[int, int], previous_state: Dict) -> bool:
        """Check if movement is directly toward food"""
        if not previous_state:
            return False
        
        prev_pos = previous_state.get('snake_position', [[0, 0]])[0]
        prev_pos = tuple(prev_pos)
        
        # Calculate if we moved closer to food in both x and y directions
        prev_x_dist = abs(prev_pos[0] - food_pos[0])
        prev_y_dist = abs(prev_pos[1] - food_pos[1])
        curr_x_dist = abs(current_pos[0] - food_pos[0])
        curr_y_dist = abs(current_pos[1] - food_pos[1])
        
        # Efficient if we reduced distance in the direction we moved
        return (curr_x_dist < prev_x_dist) or (curr_y_dist < prev_y_dist)
    
    def get_reward_summary(self) -> Dict:
        """Get summary of reward configuration"""
        return {
            "type": "Dense Reward System",
            "focus": "Immediate feedback and survival",
            "survival_reward_per_step": self.config.survival_reward_per_step,
            "food_consumption_reward": self.config.food_consumption_reward,
            "death_penalty": self.config.death_penalty,
            "wall_avoidance_reward": self.config.wall_avoidance_reward,
            "food_proximity_base": self.config.food_proximity_reward_base
        }

def test_dense_rewards():
    """Test the dense reward system with various scenarios"""
    
    print("🧪 Testing Dense Reward System")
    print("=" * 50)
    
    calculator = DenseRewardCalculator()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Survival Step",
            "action": "move",
            "state": {"snake_position": [[5, 5]], "alive": True, "foods": [[10, 10]], "grid_size": (20, 20)},
            "prev_state": {"snake_position": [[5, 4]]}
        },
        {
            "name": "Moving Toward Food",
            "action": "move", 
            "state": {"snake_position": [[6, 6]], "alive": True, "foods": [[10, 10]], "grid_size": (20, 20)},
            "prev_state": {"snake_position": [[5, 5]]}
        },
        {
            "name": "Food Consumption",
            "action": "food_eaten",
            "state": {"snake_position": [[10, 10], [9, 10]], "alive": True, "foods": [], "grid_size": (20, 20)},
            "prev_state": {"snake_position": [[10, 9]]}
        },
        {
            "name": "Wall Collision",
            "action": "collision_wall",
            "state": {"snake_position": [[0, 5]], "alive": False, "foods": [[10, 10]], "grid_size": (20, 20)},
            "prev_state": {"snake_position": [[1, 5]]}
        },
        {
            "name": "Near Wall Avoidance",
            "action": "move",
            "state": {"snake_position": [[2, 5]], "alive": True, "foods": [[10, 10]], "grid_size": (20, 20)},
            "prev_state": {"snake_position": [[1, 5]]}
        }
    ]
    
    for scenario in scenarios:
        reward = calculator.calculate_dense_reward(
            "test_snake", 
            scenario["action"], 
            scenario["state"], 
            scenario.get("prev_state")
        )
        print(f"📊 {scenario['name']}: {reward:.2f} points")
    
    print(f"\n🎯 Reward System Summary:")
    summary = calculator.get_reward_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Dense reward system tested successfully!")

if __name__ == "__main__":
    test_dense_rewards()