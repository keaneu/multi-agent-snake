"""
AI Training Agent - Neural network for autonomous snake gameplay
Implements deep Q-learning (DQN) with experience replay for training intelligent snake agents
"""

import numpy as np
import random
import time
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from multi_agent_framework import BaseAgent, MessageType, Message

# Try to import PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from enhanced_dqn import EnhancedDQN, ReplayInformedTrainer, create_enhanced_dqn
    from hrm_enhanced_dqn import HRMEnhancedDQN, HRMReplayInformedTrainer, create_hrm_enhanced_dqn
    from test_dense_rewards import DenseRewardCalculator, DenseRewardConfig
    TORCH_AVAILABLE = True
    ENHANCED_DQN_AVAILABLE = True
    HRM_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    ENHANCED_DQN_AVAILABLE = False
    HRM_AVAILABLE = False
    print(f"⚠️  PyTorch, Enhanced DQN, or HRM not available: {e}")
    print("⚠️  Using simplified Q-learning")

class Action(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

@dataclass
class TrainingConfig:
    """Configuration for AI training"""
    # Neural network parameters
    hidden_size: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    
    # Training parameters - Phase 44 Legendary Survival-First Enhancement
    epsilon_start: float = 0.1      # Start with mostly exploitation for survival learning
    epsilon_end: float = 0.05       # Low minimum for focused legendary learning
    epsilon_decay: float = 0.9999   # Ultra-slow decay for extended legendary learning
    
    # Experience replay - Phase 44 Legendary Memory Enhancement
    memory_size: int = 1000000  # Legendary memory buffer for ascension mastery
    batch_size: int = 128       # Legendary batch size for ascension learning
    target_update_freq: int = 100
    
    # Training schedule - Phase 44 Legendary Learning Enhancement
    train_freq: int = 1  # Train every step for legendary rapid learning
    save_freq: int = 50  # Save model every N episodes
    
# Phase 44: Legendary Ascension - Targeting 60-500 score range with legendary mastery
    use_dense_rewards: bool = True    # Legendary ascension reward system for legendary mastery
    food_reward: float = 5000000000.0 # Legendary food reward for ascension acquisition mastery
    death_penalty: float = 0.0        # Zero penalty maintained for legendary ascension
    step_penalty: float = 0.0         # Zero step penalty for legendary extended gameplay
    survival_reward: float = 50000000.0 # Legendary survival reward for ascension excellence
    
    # Phase 44: Legendary Ascension Rewards - 60-500 score range legendary optimization
    food_proximity_reward: float = 3000000000.0 # Legendary proximity reward for ascension targeting
    wall_avoidance_reward: float = 0.0          # Zero safety emphasis for legendary ascension
    efficient_movement_reward: float = 6000000000.0 # Legendary movement efficiency for ascension tactics
    exploration_reward: float = 1500000000.0    # Legendary exploration for ascension positioning
    length_bonus_per_segment: float = 3000000000.0 # Legendary length bonuses for ascension growth
    length_progression_reward: float = 10000000000.0 # Legendary ascension growth incentive
    safe_exploration_reward: float = 2500000000.0 # Legendary safe exploration for ascension mastery
    strategic_positioning_reward: float = 6000000000.0  # Legendary strategic positioning for ascension performance
    
    # Phase 44: Legendary Ascension Milestones - 60-500 score range legendary achievements
    legendary_60_bonus: float = 100000000000000.0   # Legendary bonus for reaching 60 points
    ascension_100_bonus: float = 500000000000000.0  # Ascension performance bonus for 100 points
    mythical_200_bonus: float = 2500000000000000.0  # Mythical mastery bonus for 200 points
    godmode_400_bonus: float = 10000000000000000.0  # Godmode performance bonus for 400+ points
    ultra_sequence_bonus: float = 40.0      # Ultra-elite bonus for championship food sequences
    mastery_growth_bonus: float = 50.0      # Ultra-elite growth milestone rewards
    ultra_patience_bonus: float = 2.0       # Ultra-elite patience for championship mastery
    
    # Ultra-Advanced Championship Bonuses - 25-40 range specialization
    territorial_supremacy_bonus: float = 3.2   # Bonus for championship territory control
    multi_food_legend_bonus: float = 4.5       # Bonus for legendary multi-food sequences
    calculated_aggression_bonus: float = 2.8   # Bonus for championship calculated risks
    championship_endurance_bonus: float = 2.4  # Bonus for maintaining championship performance
    
    # Refined Strategic Bonuses for 20-35 point consolidation
    territory_control_bonus: float = 0.8    # Moderate territory control development
    competitive_advantage_bonus: float = 1.5 # Balanced competitive positioning
    efficiency_mastery_bonus: float = 1.2   # Moderate efficiency optimization
    
    # Progressive Enhancement Bonuses - Building intermediate skills
    progressive_food_bonus: float = 0.7         # Enhanced bonus for consistent food acquisition
    tactical_movement_bonus: float = 0.4        # Bonus for tactical movement patterns
    skill_development_bonus: float = 0.3        # Bonus for skill progression milestones
    efficiency_improvement_bonus: float = 0.5   # Bonus for improving efficiency over time
    patience_bonus: float = 0.1                 # Missing attribute causing errors
    
    # Model persistence
    model_save_path: str = "snake_ai_model.pth"
    stats_save_path: str = "training_stats.json"
    memory_save_path: str = "snake_ai_memory.pkl"
    
    # HRM (Hierarchical Reasoning Model) parameters - ENABLED for breakthrough strategy
    use_hrm: bool = True
    hrm_hierarchical_loss_weight: float = 0.3
    hrm_goal_value_loss_weight: float = 0.2
    
    # Enhanced Exploration Parameters - Breakthrough Strategy
    enhanced_exploration: bool = True
    epsilon_boost: float = 0.0       # Disabled for Phase 44 survival-focused learning
    adaptive_epsilon: bool = True     # Dynamic epsilon based on performance
    plateau_threshold: int = 10       # Episodes without improvement before boost
    exploration_schedule: str = "adaptive_breakthrough"

@dataclass
class Experience:
    """Single experience for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

@dataclass
class TrainingStats:
    """Training statistics tracking"""
    episode: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    foods_eaten: int = 0
    deaths: int = 0
    epsilon: float = 1.0
    loss: float = 0.0
    q_value_avg: float = 0.0

class DQN(nn.Module):
    """Deep Q-Network for snake AI"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SimpleDQN:
    """Simplified DQN using numpy when PyTorch is not available"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        return self.z3
    
    def backward(self, x, y, output):
        m = x.shape[0]
        
        # Output layer
        dz3 = output - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

class AITrainingAgent(BaseAgent):
    """
    AI Training Agent that learns to play Snake using Deep Q-Learning
    """
    
    def __init__(self, snake_id: str, config: Optional[TrainingConfig] = None):
        super().__init__(f"ai_training_{snake_id}")
        self.snake_id = snake_id
        self.config = config or TrainingConfig()
        
        # Game state
        self.grid_width = 20
        self.grid_height = 20
        self.current_state = None
        self.last_action = None
        self.game_active = False
        
        # Snake tracking
        self.snake_segments = []
        self.snake_direction = "RIGHT"
        self.alive = True  # Snake alive status
        self.foods = []
        self.obstacles = []
        self.other_snakes = {}
        
        # Training components
        self.state_size = self.calculate_state_size()
        self.action_size = len(Action)
        
        # Initialize neural network - Phase 8: Breakthrough Strategy
        if TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE and HRM_AVAILABLE and self.config.use_hrm:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🎲🧠 Initializing Breakthrough Strategy: HRM-Enhanced DQN + Enhanced Exploration for Snake {snake_id}")
            
            # Use HRM-Enhanced DQN with replay embeddings and hierarchical reasoning
            self.q_network = create_hrm_enhanced_dqn(
                input_size=self.state_size,
                hidden_size=self.config.hidden_size,
                output_size=self.action_size
            ).to(self.device)
            
            self.target_network = create_hrm_enhanced_dqn(
                input_size=self.state_size,
                hidden_size=self.config.hidden_size,
                output_size=self.action_size
            ).to(self.device)
            
            # Use HRM replay-informed trainer
            self.trainer = HRMReplayInformedTrainer(self.q_network, self.config.learning_rate)
            self.update_target_network()
            
            # Enhanced Exploration Strategy Components
            self.exploration_boost_active = True
            self.plateau_counter = 0
            self.best_recent_score = 0
            self.exploration_episodes = 0
            self.adaptive_epsilon_multiplier = 1.0
            
            # HRM-specific tracking
            self.hierarchical_rewards = deque(maxlen=100)
            self.goal_completions = deque(maxlen=100)
            
            print(f"🎯 Breakthrough Strategy Initialized:")
            print(f"   🎲 Enhanced Exploration: {self.config.enhanced_exploration}")
            print(f"   🧠 Hierarchical Reasoning: {self.config.use_hrm}")
            print(f"   📈 Epsilon Boost: {self.config.epsilon_boost}")
            
        elif TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🧠 Initializing Enhanced DQN with replay embeddings for Snake {snake_id}")
            
            # Use Enhanced DQN with replay embeddings
            self.q_network = create_enhanced_dqn(
                input_size=self.state_size,
                hidden_size=self.config.hidden_size,
                output_size=self.action_size
            ).to(self.device)
            
            self.target_network = create_enhanced_dqn(
                input_size=self.state_size,
                hidden_size=self.config.hidden_size,
                output_size=self.action_size
            ).to(self.device)
            
            # Use replay-informed trainer
            self.trainer = ReplayInformedTrainer(self.q_network, self.config.learning_rate)
            self.update_target_network()
            
        elif TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQN(self.state_size, self.config.hidden_size, self.action_size).to(self.device)
            self.target_network = DQN(self.state_size, self.config.hidden_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
            self.update_target_network()
            self.trainer = None
        else:
            self.q_network = SimpleDQN(self.state_size, self.config.hidden_size, self.action_size, self.config.learning_rate)
            self.target_network = SimpleDQN(self.state_size, self.config.hidden_size, self.action_size, self.config.learning_rate)
            self.trainer = None
        
        # Experience replay
        self.memory = deque(maxlen=self.config.memory_size)
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.training_step = 0
        self.episode_count = 0
        
        # Statistics
        self.stats = TrainingStats()
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Timing
        self.last_decision_time = 0.0
        self.decision_interval = 0.1  # Make decisions every 100ms
        
        # Dense reward system - Phase 5: Intermediate Mastery Training
        if self.config.use_dense_rewards:
            dense_config = DenseRewardConfig(
                survival_reward_per_step=self.config.survival_reward,
                food_consumption_reward=self.config.food_reward,
                death_penalty=self.config.death_penalty,
                wall_avoidance_reward=self.config.wall_avoidance_reward,
                food_proximity_reward_base=self.config.food_proximity_reward * 3.5,
                food_proximity_multiplier=2.5,  # Enhanced proximity multiplier for precision
                efficient_movement_reward=self.config.efficient_movement_reward,
                exploration_reward=self.config.exploration_reward,
                length_bonus_per_segment=self.config.length_bonus_per_segment,
                forward_movement_reward=2.0,    # Premium movement incentive for mastery
                time_survival_bonus=0.2,        # Maximum time-based survival bonus
                streak_bonus_multiplier=1.8     # Higher streak bonuses for consistency
            )
            self.dense_reward_calculator = DenseRewardCalculator(dense_config)
            print(f"🎯 Phase 5 Intermediate Mastery dense reward system enabled for Snake {snake_id}")
        else:
            self.dense_reward_calculator = None
        
    def calculate_state_size(self) -> int:
        """Calculate the size of the state vector"""
        # State includes:
        # - Danger detection (3 directions): 3
        # - Direction encoding (4 directions): 4  
        # - Food direction relative to head: 4
        # - Distance to nearest food: 1
        # - Snake length: 1
        # - Distance to walls: 4
        # - Nearby obstacles: 8 (3x3 grid around head)
        # - Other snake proximity: 8
        return 3 + 4 + 4 + 1 + 1 + 4 + 8 + 8
    
    def initialize(self):
        """Initialize the AI training agent"""
        print(f"🤖 AI Training Agent {self.snake_id} initializing...")
        
        # Subscribe to relevant messages
        self.subscribe_to_messages([
            MessageType.GAME_STATE,
            MessageType.ENVIRONMENT_UPDATE,
            MessageType.COLLISION_EVENT,
            MessageType.FOOD_EATEN,
            MessageType.RESET_GAME
        ])
        
        # Load existing model if available
        self.load_model()
        
        # Update state
        self.state = self.get_public_state()
        
        # Announce readiness
        self.send_message(MessageType.AGENT_READY, "game_engine", {"agent_id": self.agent_id})

        print(f"✅ AI Training Agent {self.snake_id} ready - State size: {self.state_size}")
    
    def update(self):
        """Main AI update loop"""
        current_time = time.time()
        
        # Make decisions at regular intervals
        if current_time - self.last_decision_time >= self.decision_interval:
            if self.game_active and self.snake_segments:
                self.make_decision()
                self.last_decision_time = current_time
        
        # Train network periodically
        if len(self.memory) >= self.config.batch_size and self.training_step % self.config.train_freq == 0:
            self.train_network()
        
        # Update target network periodically
        if TORCH_AVAILABLE and self.training_step % self.config.target_update_freq == 0:
            self.update_target_network()
        
        # Save model periodically
        if self.episode_count > 0 and self.episode_count % self.config.save_freq == 0:
            self.save_model()
        
        # Update statistics
        self.stats.epsilon = self.epsilon
        self.state = self.get_public_state()
    
    def make_decision(self):
        """Make an AI decision for snake movement"""
        # Get current state
        current_state = self.get_game_state()
        if current_state is None:
            return
        
        # Choose action using epsilon-greedy policy
        action = self.choose_action(current_state)
        
        # Convert action to direction command
        direction = self.action_to_direction(action)
        
        # Send movement command
        self.send_message(MessageType.MOVE_COMMAND, f"snake_logic_{self.snake_id}", {
            "snake_id": self.snake_id,
            "direction": direction
        })
        
        # Store state and action for training
        self.current_state = current_state
        self.last_action = action
        
        self.training_step += 1
        
        # Calculate step reward using dense reward system
        if self.config.use_dense_rewards and self.alive:
            step_reward = self.calculate_reward("step", {})
            if step_reward > 0:  # Only add positive intermediate rewards for step
                self.stats.total_reward += step_reward
        
        # Broadcast AI decision for monitoring (less frequently to reduce spam)
        if self.training_step % 10 == 0:  # Only every 10th decision
            self.broadcast_message(MessageType.AI_DECISION, {
                "snake_id": self.snake_id,
                "action": action.name,
                "direction": direction,
                "epsilon": self.epsilon,
                "training_step": self.training_step
            })
    
    def get_game_state(self) -> Optional[np.ndarray]:
        """Extract game state features for the neural network"""
        if not self.snake_segments:
            return None
        
        head_pos = self.snake_segments[0]
        state_features = []
        
        # 1. Danger detection in 3 directions (straight, left, right)
        dangers = self.get_danger_detection(head_pos)
        state_features.extend(dangers)
        
        # 2. Current direction (one-hot encoding)
        direction_encoding = self.encode_direction(self.snake_direction)
        state_features.extend(direction_encoding)
        
        # 3. Food direction relative to head
        food_direction = self.get_food_direction(head_pos)
        state_features.extend(food_direction)
        
        # 4. Distance to nearest food (normalized)
        food_distance = self.get_nearest_food_distance(head_pos)
        state_features.append(food_distance)
        
        # 5. Snake length (normalized)
        length_normalized = len(self.snake_segments) / (self.grid_width * self.grid_height)
        state_features.append(length_normalized)
        
        # 6. Distance to walls (normalized)
        wall_distances = self.get_wall_distances(head_pos)
        state_features.extend(wall_distances)
        
        # 7. Nearby obstacles (3x3 grid around head)
        nearby_obstacles = self.get_nearby_obstacles(head_pos)
        state_features.extend(nearby_obstacles)
        
        # 8. Other snake proximity
        other_snake_proximity = self.get_other_snake_proximity(head_pos)
        state_features.extend(other_snake_proximity)
        
        return np.array(state_features, dtype=np.float32)
    
    def get_danger_detection(self, head_pos: Tuple[int, int]) -> List[float]:
        """Detect danger in straight, left, and right directions"""
        x, y = head_pos
        
        # Get direction vectors
        direction_map = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0)
        }
        
        dx, dy = direction_map[self.snake_direction]
        
        # Calculate positions for straight, left, right
        straight_pos = (x + dx, y + dy)
        left_dx, left_dy = self.get_left_direction(dx, dy)
        left_pos = (x + left_dx, y + left_dy)
        right_dx, right_dy = self.get_right_direction(dx, dy)
        right_pos = (x + right_dx, y + right_dy)
        
        positions = [straight_pos, left_pos, right_pos]
        dangers = []
        
        for pos in positions:
            danger = 0.0
            
            # Check wall collision
            if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
                danger = 1.0
            # Check self collision
            elif pos in self.snake_segments:
                danger = 1.0
            # Check obstacle collision
            elif pos in self.obstacles:
                danger = 1.0
            # Check other snake collision
            else:
                for other_segments in self.other_snakes.values():
                    if pos in other_segments:
                        danger = 1.0
                        break
            
            dangers.append(danger)
        
        return dangers
    
    def get_left_direction(self, dx: int, dy: int) -> Tuple[int, int]:
        """Get left direction vector"""
        if dx == 0 and dy == -1:  # UP
            return (-1, 0)  # LEFT
        elif dx == 0 and dy == 1:  # DOWN
            return (1, 0)   # RIGHT
        elif dx == -1 and dy == 0:  # LEFT
            return (0, 1)   # DOWN
        elif dx == 1 and dy == 0:   # RIGHT
            return (0, -1)  # UP
        return (0, 0)
    
    def get_right_direction(self, dx: int, dy: int) -> Tuple[int, int]:
        """Get right direction vector"""
        if dx == 0 and dy == -1:  # UP
            return (1, 0)   # RIGHT
        elif dx == 0 and dy == 1:  # DOWN
            return (-1, 0)  # LEFT
        elif dx == -1 and dy == 0:  # LEFT
            return (0, -1)  # UP
        elif dx == 1 and dy == 0:   # RIGHT
            return (0, 1)   # DOWN
        return (0, 0)
    
    def encode_direction(self, direction: str) -> List[float]:
        """Encode direction as one-hot vector"""
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        encoding = [0.0] * 4
        if direction in directions:
            encoding[directions.index(direction)] = 1.0
        return encoding
    
    def get_food_direction(self, head_pos: Tuple[int, int]) -> List[float]:
        """Get direction to nearest food"""
        if not self.foods:
            return [0.0, 0.0, 0.0, 0.0]  # No food
        
        x, y = head_pos
        nearest_food = min(self.foods, key=lambda f: abs(f[0] - x) + abs(f[1] - y))
        
        food_x, food_y = nearest_food
        
        # Calculate relative direction
        up = 1.0 if food_y < y else 0.0
        down = 1.0 if food_y > y else 0.0
        left = 1.0 if food_x < x else 0.0
        right = 1.0 if food_x > x else 0.0
        
        return [up, down, left, right]
    
    def get_nearest_food_distance(self, head_pos: Tuple[int, int]) -> float:
        """Get distance to nearest food (normalized)"""
        if not self.foods:
            return 1.0  # Maximum distance when no food
        
        x, y = head_pos
        min_distance = min(abs(f[0] - x) + abs(f[1] - y) for f in self.foods)
        max_possible_distance = self.grid_width + self.grid_height
        
        return min_distance / max_possible_distance
    
    def get_wall_distances(self, head_pos: Tuple[int, int]) -> List[float]:
        """Get normalized distances to walls in 4 directions"""
        x, y = head_pos
        
        up_dist = y / self.grid_height
        down_dist = (self.grid_height - 1 - y) / self.grid_height
        left_dist = x / self.grid_width
        right_dist = (self.grid_width - 1 - x) / self.grid_width
        
        return [up_dist, down_dist, left_dist, right_dist]
    
    def get_nearby_obstacles(self, head_pos: Tuple[int, int]) -> List[float]:
        """Get obstacles in 3x3 grid around head"""
        x, y = head_pos
        obstacles_grid = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip center (head position)
                    continue
                
                check_pos = (x + dx, y + dy)
                
                # Check if position has obstacle
                has_obstacle = 0.0
                if check_pos in self.obstacles:
                    has_obstacle = 1.0
                elif check_pos in self.snake_segments:
                    has_obstacle = 1.0
                elif (check_pos[0] < 0 or check_pos[0] >= self.grid_width or 
                      check_pos[1] < 0 or check_pos[1] >= self.grid_height):
                    has_obstacle = 1.0
                
                obstacles_grid.append(has_obstacle)
        
        return obstacles_grid
    
    def get_other_snake_proximity(self, head_pos: Tuple[int, int]) -> List[float]:
        """Get proximity to other snakes in 8 directions"""
        x, y = head_pos
        proximities = []
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            proximity = 0.0
            
            # Check positions in this direction
            for distance in range(1, 4):  # Check up to 3 cells away
                check_pos = (x + dx * distance, y + dy * distance)
                
                # Check if any other snake is at this position
                for other_segments in self.other_snakes.values():
                    if check_pos in other_segments:
                        proximity = 1.0 / distance  # Closer = higher value
                        break
                
                if proximity > 0:
                    break
            
            proximities.append(proximity)
        
        return proximities
    
    def calculate_intermediate_rewards(self) -> float:
        """Calculate dense intermediate rewards for continuous learning"""
        if not self.snake_segments:
            return 0.0
            
        total_intermediate_reward = 0.0
        head_pos = self.snake_segments[0]
        
        # 1. Food Proximity Reward - reward for getting closer to food
        food_proximity_reward = self.calculate_food_proximity_reward(head_pos)
        total_intermediate_reward += food_proximity_reward
        
        # 2. Wall Avoidance Reward - reward for maintaining safe distance from walls
        wall_avoidance_reward = self.calculate_wall_avoidance_reward(head_pos)
        total_intermediate_reward += wall_avoidance_reward
        
        # 3. Efficient Movement Reward - reward for not moving in circles
        efficient_movement_reward = self.calculate_efficient_movement_reward(head_pos)
        total_intermediate_reward += efficient_movement_reward
        
        # 4. Safe Exploration Reward - reward for exploring new areas safely
        safe_exploration_reward = self.calculate_safe_exploration_reward(head_pos)
        total_intermediate_reward += safe_exploration_reward
        
        # 5. Strategic Positioning Reward - reward for good positioning relative to other snakes
        strategic_positioning_reward = self.calculate_strategic_positioning_reward(head_pos)
        total_intermediate_reward += strategic_positioning_reward
        
        return total_intermediate_reward
    
    def calculate_food_proximity_reward(self, head_pos: Tuple[int, int]) -> float:
        """Reward for moving closer to food"""
        if not self.foods:
            return 0.0
            
        # Find nearest food
        x, y = head_pos
        nearest_food = min(self.foods, key=lambda f: abs(f[0] - x) + abs(f[1] - y))
        current_distance = abs(nearest_food[0] - x) + abs(nearest_food[1] - y)
        
        # Store previous distance if not exists
        if not hasattr(self, '_prev_food_distance'):
            self._prev_food_distance = current_distance
            return 0.0
            
        # Reward for getting closer, penalty for getting farther
        distance_change = self._prev_food_distance - current_distance
        self._prev_food_distance = current_distance
        
        # Scale reward based on improvement
        proximity_reward = distance_change * self.config.food_proximity_reward
        
        # Enhanced proximity bonuses for 10-25 point range
        if current_distance <= 1:
            proximity_reward += self.config.food_proximity_reward * 1.0  # Strong bonus for adjacent
        elif current_distance <= 2:
            proximity_reward += self.config.food_proximity_reward * 0.5  # Medium bonus for close
            
        return proximity_reward
    
    def calculate_wall_avoidance_reward(self, head_pos: Tuple[int, int]) -> float:
        """Reward for maintaining safe distance from walls"""
        x, y = head_pos
        
        # Calculate minimum distance to any wall
        min_wall_distance = min(
            x,  # distance to left wall
            y,  # distance to top wall
            self.grid_width - 1 - x,  # distance to right wall
            self.grid_height - 1 - y   # distance to bottom wall
        )
        
        # Reward for maintaining safe distance (at least 2 cells from wall)
        if min_wall_distance >= 3:
            return self.config.wall_avoidance_reward
        elif min_wall_distance >= 2:
            return self.config.wall_avoidance_reward * 0.5
        elif min_wall_distance >= 1:
            return self.config.wall_avoidance_reward * 0.1
        else:
            return -self.config.wall_avoidance_reward  # Penalty for being too close
    
    def calculate_efficient_movement_reward(self, head_pos: Tuple[int, int]) -> float:
        """Reward for efficient movement patterns (avoiding loops)"""
        # Track recent positions to detect loops
        if not hasattr(self, '_recent_positions'):
            self._recent_positions = deque(maxlen=8)
            
        self._recent_positions.append(head_pos)
        
        if len(self._recent_positions) < 4:
            return 0.0
            
        # Penalty for visiting same position recently
        position_counts = {}
        for pos in self._recent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
            
        max_visits = max(position_counts.values())
        
        if max_visits >= 3:  # Visiting same position 3+ times recently
            return -self.config.efficient_movement_reward
        elif max_visits >= 2:  # Visiting same position twice
            return -self.config.efficient_movement_reward * 0.5
        else:
            return self.config.efficient_movement_reward * 0.3  # Small reward for good movement
    
    def calculate_safe_exploration_reward(self, head_pos: Tuple[int, int]) -> float:
        """Reward for exploring new areas while maintaining safety"""
        x, y = head_pos
        
        # Track explored positions
        if not hasattr(self, '_explored_positions'):
            self._explored_positions = set()
            
        # Check if this is a new position
        is_new_position = head_pos not in self._explored_positions
        self._explored_positions.add(head_pos)
        
        if not is_new_position:
            return 0.0
            
        # Check safety of current position (no immediate dangers)
        dangers = self.get_danger_detection(head_pos)
        danger_count = sum(dangers)
        
        # Reward for safe exploration
        if danger_count == 0:  # No dangers
            return self.config.safe_exploration_reward
        elif danger_count == 1:  # One danger (manageable)
            return self.config.safe_exploration_reward * 0.5
        else:  # Multiple dangers
            return 0.0
    
    def calculate_strategic_positioning_reward(self, head_pos: Tuple[int, int]) -> float:
        """Reward for good strategic positioning relative to other snakes"""
        if not self.other_snakes:
            return 0.0
            
        x, y = head_pos
        total_positioning_reward = 0.0
        
        for other_snake_segments in self.other_snakes.values():
            if not other_snake_segments:
                continue
                
            other_head = other_snake_segments[0]
            other_x, other_y = other_head
            
            # Distance to other snake's head
            distance_to_other = abs(other_x - x) + abs(other_y - y)
            
            # Reward for maintaining optimal distance (not too close, not too far)
            if 4 <= distance_to_other <= 8:  # Optimal distance range
                total_positioning_reward += self.config.strategic_positioning_reward
            elif 2 <= distance_to_other < 4:  # Close but manageable
                total_positioning_reward += self.config.strategic_positioning_reward * 0.3
            elif distance_to_other < 2:  # Too close - dangerous
                total_positioning_reward -= self.config.strategic_positioning_reward
                
            # Bonus for being longer than other snake
            if len(self.snake_segments) > len(other_snake_segments):
                total_positioning_reward += self.config.strategic_positioning_reward * 0.2
                
        return total_positioning_reward
    
    def _calculate_territory_control_bonus(self) -> float:
        """Calculate bonus for controlling board territory (20-35 point range)"""
        if not self.snake_segments or len(self.snake_segments) < 12:
            return 0.0
            
        head_x, head_y = self.snake_segments[0]
        
        # Calculate controlled area based on snake body coverage
        body_positions = set(self.snake_segments)
        control_radius = min(3, len(self.snake_segments) // 8)  # Dynamic radius based on size
        
        controlled_cells = 0
        total_cells_in_radius = 0
        
        for dx in range(-control_radius, control_radius + 1):
            for dy in range(-control_radius, control_radius + 1):
                nx, ny = head_x + dx, head_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    total_cells_in_radius += 1
                    # Check if this area is "controlled" (near snake body or food-accessible)
                    min_distance_to_body = min(abs(nx - bx) + abs(ny - by) for bx, by in body_positions)
                    if min_distance_to_body <= 2:  # Within 2 cells of snake body
                        controlled_cells += 1
        
        control_ratio = controlled_cells / max(total_cells_in_radius, 1)
        return control_ratio * self.config.territory_control_bonus
    
    def _calculate_competitive_advantage_bonus(self) -> float:
        """Calculate bonus for outperforming opponent (25-40 point range)"""
        if not self.other_snakes:
            return 0.0
            
        my_length = len(self.snake_segments)
        competitor_lengths = [len(segments) for segments in self.other_snakes.values() if segments]
        
        if not competitor_lengths:
            return 0.0
            
        max_competitor_length = max(competitor_lengths)
        avg_competitor_length = sum(competitor_lengths) / len(competitor_lengths)
        
        # Bonus for being significantly longer than competitors
        if my_length > max_competitor_length + 3:
            return self.config.competitive_advantage_bonus * 1.0  # Full bonus
        elif my_length > avg_competitor_length + 2:
            return self.config.competitive_advantage_bonus * 0.5  # Half bonus
        else:
            return 0.0
    
    def _calculate_efficiency_mastery_bonus(self) -> float:
        """Calculate bonus for optimal movement patterns (25-40 point range)"""
        if not hasattr(self, '_movement_efficiency_history'):
            self._movement_efficiency_history = deque(maxlen=20)
            return 0.0
            
        # Track recent movement efficiency
        if hasattr(self, '_prev_head_pos') and self.snake_segments:
            current_head = self.snake_segments[0]
            prev_head = self._prev_head_pos
            
            # Calculate movement efficiency (avoid backtracking, prefer straight lines)
            if hasattr(self, '_prev_prev_head_pos'):
                # Check for smooth movement patterns
                dx1 = current_head[0] - prev_head[0]
                dy1 = current_head[1] - prev_head[1]
                dx2 = prev_head[0] - self._prev_prev_head_pos[0]
                dy2 = prev_head[1] - self._prev_prev_head_pos[1]
                
                # Reward consistent direction
                if (dx1, dy1) == (dx2, dy2):  # Same direction
                    efficiency_score = 1.0
                elif abs(dx1) + abs(dy1) == 1 and abs(dx2) + abs(dy2) == 1:  # Valid movement
                    efficiency_score = 0.5
                else:
                    efficiency_score = 0.0
                    
                self._movement_efficiency_history.append(efficiency_score)
                
            self._prev_prev_head_pos = prev_head
            
        self._prev_head_pos = self.snake_segments[0] if self.snake_segments else None
        
        # Calculate recent efficiency average
        if len(self._movement_efficiency_history) >= 10:
            avg_efficiency = sum(self._movement_efficiency_history) / len(self._movement_efficiency_history)
            if avg_efficiency > 0.7:  # High efficiency threshold
                return self.config.efficiency_mastery_bonus * avg_efficiency
                
        return 0.0
    
    def _calculate_progressive_food_bonus(self) -> float:
        """Calculate enhanced bonus for consistent food acquisition (Progressive Enhancement)"""
        if not hasattr(self, '_food_acquisition_streak'):
            self._food_acquisition_streak = 0
            
        current_length = len(self.snake_segments)
        if hasattr(self, '_prev_length') and current_length > self._prev_length:
            # Food was acquired - increment streak and give progressive bonus
            self._food_acquisition_streak += 1
            base_bonus = self.config.progressive_food_bonus
            # Progressive bonus increases with streak (capped at 3x)
            streak_multiplier = min(1.0 + (self._food_acquisition_streak - 1) * 0.2, 3.0)
            return base_bonus * streak_multiplier
        else:
            # Reset streak if no food acquired this step
            if hasattr(self, '_prev_length'):
                self._food_acquisition_streak = max(0, self._food_acquisition_streak - 0.1)
                
        self._prev_length = current_length
        return 0.0
    
    def _calculate_tactical_movement_bonus(self) -> float:
        """Calculate bonus for tactical movement patterns (Progressive Enhancement)"""
        if not hasattr(self, '_movement_history'):
            self._movement_history = deque(maxlen=8)
            return 0.0
            
        if self.alive and self.snake_segments and len(self.snake_segments) > 0:
            head_x, head_y = self.snake_segments[0]
            
            # Track movement direction
            if hasattr(self, '_prev_head_pos'):
                prev_x, prev_y = self._prev_head_pos
                move_direction = (head_x - prev_x, head_y - prev_y)
                self._movement_history.append(move_direction)
            
            self._prev_head_pos = (head_x, head_y)
            
            # Reward tactical movement patterns (not just random movement)
            if len(self._movement_history) >= 4:
                # Check for purposeful movement (towards food, away from walls)
                recent_moves = list(self._movement_history)[-4:]
                direction_variance = len(set(recent_moves))
                
                # Reward moderate variance (not too erratic, not too repetitive)
                if 2 <= direction_variance <= 3:
                    return self.config.tactical_movement_bonus
                    
        return 0.0
    
    def _calculate_skill_development_bonus(self) -> float:
        """Calculate bonus for skill progression milestones (Progressive Enhancement)"""
        if not hasattr(self, '_skill_milestones'):
            self._skill_milestones = {'max_length': 3, 'survival_time': 0}
            
        current_length = len(self.snake_segments)
        current_time = getattr(self, 'move_count', 0)
        
        bonus = 0.0
        
        # Length milestone bonus
        if current_length > self._skill_milestones['max_length']:
            bonus += self.config.skill_development_bonus * 0.5
            self._skill_milestones['max_length'] = current_length
            
        # Survival time milestone bonus
        if current_time > self._skill_milestones['survival_time'] + 50:  # Every 50 steps
            bonus += self.config.skill_development_bonus * 0.3
            self._skill_milestones['survival_time'] = current_time
            
        return bonus
    
    def _calculate_efficiency_improvement_bonus(self) -> float:
        """Calculate bonus for improving efficiency over time (Progressive Enhancement)"""
        if not hasattr(self, '_efficiency_tracking'):
            self._efficiency_tracking = {'moves_per_food': deque(maxlen=5), 'recent_efficiency': 0.0}
            return 0.0
            
        current_length = len(self.snake_segments)
        current_moves = getattr(self, 'move_count', 0)
        
        # Track efficiency when food is acquired
        if hasattr(self, '_prev_length') and current_length > self._prev_length:
            moves_since_start = current_moves
            if moves_since_start > 0:
                efficiency = 1.0 / moves_since_start  # Higher efficiency = fewer moves per food
                self._efficiency_tracking['moves_per_food'].append(efficiency)
                
                # Calculate improvement bonus
                if len(self._efficiency_tracking['moves_per_food']) >= 3:
                    recent_avg = sum(list(self._efficiency_tracking['moves_per_food'])[-2:]) / 2
                    older_avg = sum(list(self._efficiency_tracking['moves_per_food'])[:-2]) / max(1, len(self._efficiency_tracking['moves_per_food']) - 2)
                    
                    if recent_avg > older_avg:  # Efficiency improved
                        improvement = (recent_avg - older_avg) / older_avg
                        return min(improvement * self.config.efficiency_improvement_bonus, self.config.efficiency_improvement_bonus)
        
        self._prev_length = current_length
        return 0.0
    
    def reset_intermediate_reward_tracking(self):
        """Reset tracking variables for intermediate rewards at episode start"""
        # Reset food proximity tracking
        if hasattr(self, '_prev_food_distance'):
            delattr(self, '_prev_food_distance')
            
        # Reset movement efficiency tracking
        if hasattr(self, '_recent_positions'):
            self._recent_positions.clear()
            
        # Reset exploration tracking every few episodes to encourage re-exploration
        if hasattr(self, '_explored_positions') and self.episode_count % 20 == 0:
            self._explored_positions.clear()
    
    def choose_action(self, state: np.ndarray) -> Action:
        """Choose action using enhanced exploration epsilon-greedy policy with HRM hierarchical reasoning"""
        
        # Enhanced Exploration Strategy - Apply epsilon boost if configured
        effective_epsilon = self.epsilon
        if (hasattr(self.config, 'enhanced_exploration') and self.config.enhanced_exploration and
            hasattr(self, 'exploration_boost_active') and self.exploration_boost_active):
            
            # Apply adaptive epsilon boost for breakthrough discovery
            boost_factor = self.config.epsilon_boost * self.adaptive_epsilon_multiplier
            effective_epsilon = min(0.8, self.epsilon + boost_factor)  # Cap at 80% exploration
            
            # Adaptive exploration based on plateau detection
            if hasattr(self, 'plateau_counter') and self.plateau_counter > self.config.plateau_threshold:
                effective_epsilon = min(0.9, effective_epsilon * 1.5)  # Extra boost during plateau
        
        if random.random() < effective_epsilon:
            return random.choice(list(Action))
        else:
            # Use HRM-enhanced action selection if available
            if (hasattr(self, 'q_network') and 
                hasattr(self.q_network, 'get_action_with_hrm_explanation') and 
                TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE and HRM_AVAILABLE and self.config.use_hrm):
                
                state_tensor = torch.FloatTensor(state).to(self.device)
                action_idx, explanation = self.q_network.get_action_with_hrm_explanation(state_tensor, epsilon=0.0)
                
                # Log HRM decision explanation periodically
                if self.training_step % 500 == 0:
                    print(f"🧠🏗️ Snake {self.snake_id} HRM Decision: Action {action_idx}")
                    print(f"   Decision type: {explanation['decision_type']}")
                    print(f"   Active goals: {explanation.get('active_goals', [])}")
                    print(f"   HRM confidence: {explanation.get('dqn_confidence', 0):.3f}")
                    print(f"   Active option: {explanation.get('active_option', 'None')}")
                
                # Track hierarchical rewards if available
                if hasattr(self.q_network, 'hrm') and hasattr(self, 'hierarchical_rewards'):
                    hrm_metrics = self.q_network.get_hrm_metrics()
                    if 'hierarchical_reward' in hrm_metrics:
                        self.hierarchical_rewards.append(hrm_metrics['hierarchical_reward'])
                
                return Action(action_idx)
            
            # Fallback to enhanced DQN action selection
            elif (hasattr(self, 'q_network') and 
                  hasattr(self.q_network, 'get_action_with_explanation') and 
                  TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE):
                
                state_tensor = torch.FloatTensor(state).to(self.device)
                action_idx, explanation = self.q_network.get_action_with_explanation(state_tensor, epsilon=0.0)
                
                # Log decision explanation periodically
                if self.training_step % 500 == 0:
                    print(f"🧠 Snake {self.snake_id} Enhanced Decision: Action {action_idx}, "
                          f"Confidence: {explanation['confidence']:.3f}, "
                          f"Replay-informed: {explanation['replay_informed']}")
                
                return Action(action_idx)
            else:
                # Standard Q-value selection
                q_values = self.get_q_values(state)
                return Action(np.argmax(q_values))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for given state"""
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.cpu().numpy().flatten()
        else:
            return self.q_network.forward(state).flatten()
    
    def action_to_direction(self, action: Action) -> str:
        """Convert action to direction string"""
        direction_map = {
            "UP": {"STRAIGHT": "UP", "LEFT": "LEFT", "RIGHT": "RIGHT"},
            "DOWN": {"STRAIGHT": "DOWN", "LEFT": "RIGHT", "RIGHT": "LEFT"},
            "LEFT": {"STRAIGHT": "LEFT", "LEFT": "DOWN", "RIGHT": "UP"},
            "RIGHT": {"STRAIGHT": "RIGHT", "LEFT": "UP", "RIGHT": "DOWN"}
        }
        
        action_names = {Action.STRAIGHT: "STRAIGHT", Action.LEFT: "LEFT", Action.RIGHT: "RIGHT"}
        action_name = action_names[action]
        
        return direction_map[self.snake_direction][action_name]
    
    def calculate_reward(self, event_type: str, data: Dict[str, Any]) -> float:
        """Calculate reward using dense reward system for immediate feedback"""
        
        # Use dense reward system if enabled
        if self.config.use_dense_rewards and self.dense_reward_calculator:
            # Prepare game state for dense reward calculation
            game_state = {
                'snake_position': self.snake_segments,
                'alive': self.alive,
                'foods': self.foods,
                'grid_size': (self.grid_width, self.grid_height)
            }
            
            # Get previous state if available
            previous_state = None
            if hasattr(self, '_previous_game_state'):
                previous_state = self._previous_game_state
            
            # Calculate dense reward
            dense_reward = self.dense_reward_calculator.calculate_dense_reward(
                self.snake_id, event_type, game_state, previous_state
            )
            
            # Store current state for next calculation
            self._previous_game_state = game_state.copy()
            
            return dense_reward
        
        # Fallback to original reward system
        base_reward = 0.0
        
        if event_type == "food_eaten":
            base_reward = self.config.food_reward
            
        elif event_type == "collision":
            base_reward = self.config.death_penalty
            
        elif event_type == "step":
            base_reward = self.config.step_penalty
            
        elif event_type == "survival":
            base_reward = self.config.survival_reward
            
        # Add dense intermediate rewards for survival events
        if event_type in ["survival", "step"] and self.snake_segments:
            intermediate_rewards += self.calculate_intermediate_rewards()
            
            # NEW: Progressive Enhancement bonuses for 15-30 point range
            if len(self.snake_segments) >= 3:  # For progressive gameplay
                # Tactical movement bonus
                movement_bonus = self._calculate_tactical_movement_bonus()
                intermediate_rewards += movement_bonus
                
                # Skill development bonus
                skill_bonus = self._calculate_skill_development_bonus()
                intermediate_rewards += skill_bonus
                
                # Efficiency improvement bonus
                efficiency_bonus = self._calculate_efficiency_improvement_bonus()
                intermediate_rewards += efficiency_bonus
            
            # Debug logging for intermediate rewards (every 50 steps)
            if self.training_step % 50 == 0 and intermediate_rewards != 0:
                print(f"🎯 Snake {self.snake_id} intermediate rewards: {intermediate_rewards:.3f} (base: {base_reward:.3f})")
        
        total_reward = base_reward + intermediate_rewards
        
        # Add HRM hierarchical reward if available
        if (hasattr(self, 'q_network') and hasattr(self.q_network, 'update_hrm_rewards') and
            TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE and HRM_AVAILABLE and self.config.use_hrm):
            
            done = (event_type == "collision")
            hierarchical_bonus = self.q_network.update_hrm_rewards(total_reward, done)
            
            # Track hierarchical rewards for monitoring
            if hasattr(self, 'hierarchical_rewards'):
                self.hierarchical_rewards.append(hierarchical_bonus)
            
            return total_reward + hierarchical_bonus
        
        return total_reward
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def train_network(self):
        """Train the neural network using experience replay"""
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.config.batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        if TORCH_AVAILABLE:
            self.train_pytorch_network(states, actions, rewards, next_states, dones)
        else:
            self.train_simple_network(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def train_pytorch_network(self, states, actions, rewards, next_states, dones):
        """Train PyTorch neural network"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Use enhanced trainer if available
        if hasattr(self, 'trainer') and self.trainer is not None:
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones, self.config.gamma)
            self.stats.loss = loss
            
            # Periodically update replay patterns
            if self.training_step % 1000 == 0:
                self.trainer.update_replay_patterns()
        else:
            # Fallback to standard training
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.stats.loss = loss.item()
    
    def train_simple_network(self, states, actions, rewards, next_states, dones):
        """Train simple numpy neural network"""
        # Get current Q values
        current_q_values = np.array([self.q_network.forward(state) for state in states])
        
        # Get next Q values from target network
        next_q_values = np.array([self.target_network.forward(state) for state in next_states])
        
        # Create target Q values
        target_q_values = current_q_values.copy()
        
        for i in range(len(states)):  # Fixed: was len(batch) which is undefined
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.config.gamma * np.max(next_q_values[i])
        
        # Train network
        for i in range(len(states)):
            state = states[i].reshape(1, -1)
            target = target_q_values[i].reshape(1, -1)
            output = self.q_network.forward(state)
            self.q_network.backward(state, target, output)
    
    def update_target_network(self):
        """Update target network with current network weights"""
        if TORCH_AVAILABLE:
            try:
                self.target_network.load_state_dict(self.q_network.state_dict())
            except RuntimeError as e:
                if "unexpected key" in str(e).lower() or "missing key" in str(e).lower():
                    # Handle dynamic layer mismatch - copy compatible layers only
                    print(f"⚠️  Target network update skipped due to dynamic layers: {str(e)[:100]}...")
                    self._update_target_network_partial()
                else:
                    raise e
        else:
            # Copy weights for simple network
            self.target_network.W1 = self.q_network.W1.copy()
            self.target_network.b1 = self.q_network.b1.copy()
            self.target_network.W2 = self.q_network.W2.copy()
            self.target_network.b2 = self.q_network.b2.copy()
            self.target_network.W3 = self.q_network.W3.copy()
            self.target_network.b3 = self.q_network.b3.copy()
    
    def _update_target_network_partial(self):
        """Partially update target network, skipping dynamic layers"""
        try:
            # Get state dicts
            source_state = self.q_network.state_dict()
            target_state = self.target_network.state_dict()
            
            # Copy only compatible layers
            compatible_layers = []
            for key in source_state.keys():
                if key in target_state and source_state[key].shape == target_state[key].shape:
                    target_state[key] = source_state[key].clone()
                    compatible_layers.append(key)
            
            # Load the partially updated state dict
            self.target_network.load_state_dict(target_state, strict=False)
            
            if len(compatible_layers) > 0:
                print(f"✅ Partial target network update: {len(compatible_layers)} layers copied")
                
        except Exception as e:
            print(f"⚠️  Partial target network update failed: {e}")
            # Skip this update cycle
    
    def save_model(self):
        """Save the trained model"""
        # Prevent excessive saving - only save once per episode
        if hasattr(self, '_last_save_episode') and self._last_save_episode == self.episode_count:
            return
        
        try:
            if TORCH_AVAILABLE:
                # Handle different optimizer configurations (DQN vs HRM)
                optimizer_state = None
                if hasattr(self, 'optimizer'):
                    optimizer_state = self.optimizer.state_dict()
                elif hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizer'):
                    optimizer_state = self.trainer.optimizer.state_dict()
                
                save_data = {
                    'model_state_dict': self.q_network.state_dict(),
                    'epsilon': self.epsilon,
                    'episode': self.episode_count,
                    'stats': self.stats
                }
                
                if optimizer_state is not None:
                    save_data['optimizer_state_dict'] = optimizer_state
                
                torch.save(save_data, self.config.model_save_path)
            else:
                # Save simple network weights
                model_data = {
                    'W1': self.q_network.W1.tolist(),
                    'b1': self.q_network.b1.tolist(),
                    'W2': self.q_network.W2.tolist(),
                    'b2': self.q_network.b2.tolist(),
                    'W3': self.q_network.W3.tolist(),
                    'b3': self.q_network.b3.tolist(),
                    'epsilon': self.epsilon,
                    'episode': self.episode_count
                }
                with open(self.config.model_save_path, 'w') as f:
                    json.dump(model_data, f)
            
            self._last_save_episode = self.episode_count
            print(f"💾 Model saved for {self.snake_id} at episode {self.episode_count}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def load_model(self):
        """Load existing model if available"""
        try:
            if TORCH_AVAILABLE and os.path.exists(self.config.model_save_path):
                # Fix for PyTorch 2.6+ weights_only default change
                checkpoint = torch.load(self.config.model_save_path, map_location=self.device, weights_only=False)
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
                
                # Handle different optimizer configurations (DQN vs HRM)
                if 'optimizer_state_dict' in checkpoint:
                    if hasattr(self, 'optimizer'):
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    elif hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizer'):
                        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
                self.episode_count = checkpoint.get('episode', 0)
                self.update_target_network()
                print(f"📂 Loaded PyTorch model for {self.snake_id}")
                
            elif not TORCH_AVAILABLE and os.path.exists(self.config.model_save_path):
                with open(self.config.model_save_path, 'r') as f:
                    model_data = json.load(f)
                
                self.q_network.W1 = np.array(model_data['W1'])
                self.q_network.b1 = np.array(model_data['b1'])
                self.q_network.W2 = np.array(model_data['W2'])
                self.q_network.b2 = np.array(model_data['b2'])
                self.q_network.W3 = np.array(model_data['W3'])
                self.q_network.b3 = np.array(model_data['b3'])
                self.epsilon = model_data.get('epsilon', self.config.epsilon_start)
                self.episode_count = model_data.get('episode', 0)
                self.update_target_network()
                print(f"📂 Loaded simple model for {self.snake_id}")
                
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
    
    def process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.GAME_STATE:
                self.handle_game_state_update(message.data)
                
            elif message.type == MessageType.ENVIRONMENT_UPDATE:
                self.handle_environment_update(message.data)
                
            elif message.type == MessageType.COLLISION_EVENT:
                self.handle_collision_event(message.data)
                
            elif message.type == MessageType.FOOD_EATEN:
                self.handle_food_eaten(message.data)
                
            elif message.type == MessageType.RESET_GAME:
                self.handle_game_reset(message.data)
                
        except Exception as e:
            print(f"❌ AI Training error processing message: {e}")
    
    def handle_game_state_update(self, data: Dict[str, Any]):
        """Handle game state updates"""
        update_type = data.get("type")
        
        if update_type == "config":
            config = data.get("config", {})
            self.grid_width = config.get("grid_width", self.grid_width)
            self.grid_height = config.get("grid_height", self.grid_height)
            
        elif update_type == "update":
            phase = data.get("phase", "")
            self.game_active = (phase == "running")
            
        elif update_type == "snake_update":
            snake_id = data.get("snake_id")
            
            if snake_id == self.snake_id:
                self.snake_segments = data.get("segments", [])
                direction = data.get("direction", "RIGHT")
                self.snake_direction = direction
                
                # Calculate reward for survival
                if self.current_state is not None and self.game_active:
                    reward = self.calculate_reward("survival", {})
                    self.stats.total_reward += reward
                    
                    # Store experience if we have a complete transition
                    current_state = self.get_game_state()
                    if current_state is not None and self.last_action is not None:
                        self.store_experience(
                            self.current_state,
                            self.last_action.value,
                            reward,
                            current_state,
                            False
                        )
    
    def handle_environment_update(self, data: Dict[str, Any]):
        """Handle environment updates"""
        self.foods = data.get("foods", [])
        self.obstacles = data.get("obstacles", [])
        
        # Update other snakes
        snakes = data.get("snakes", {})
        self.other_snakes = {
            snake_id: segments for snake_id, segments in snakes.items()
            if snake_id != self.snake_id
        }
    
    def handle_collision_event(self, data: Dict[str, Any]):
        """Handle collision events"""
        player_id = data.get("player_id")
        
        if player_id == self.snake_id:
            # Mark snake as dead
            self.alive = False
            
            # Calculate death penalty
            reward = self.calculate_reward("collision", data)
            self.stats.total_reward += reward
            self.stats.deaths += 1
            
            # Store final experience
            if self.current_state is not None and self.last_action is not None:
                current_state = self.get_game_state()
                if current_state is None:
                    current_state = self.current_state  # Use last known state
                
                self.store_experience(
                    self.current_state,
                    self.last_action.value,
                    reward,
                    current_state,
                    True  # Episode is done
                )
            
            # End episode
            self.end_episode()
    
    def handle_food_eaten(self, data: Dict[str, Any]):
        """Handle food eaten events"""
        player_id = data.get("player_id")
        
        if player_id == self.snake_id:
            # Calculate food reward
            reward = self.calculate_reward("food_eaten", data)
            self.stats.total_reward += reward
            self.stats.foods_eaten += 1
            
            # Store experience
            if self.current_state is not None and self.last_action is not None:
                current_state = self.get_game_state()
                if current_state is not None:
                    self.store_experience(
                        self.current_state,
                        self.last_action.value,
                        reward,
                        current_state,
                        False
                    )
    
    def handle_game_reset(self, data: Dict[str, Any]):
        """Handle game reset"""
        # Update configuration if provided
        config = data.get("config", {})
        if config:
            self.grid_width = config.get("grid_width", self.grid_width)
            self.grid_height = config.get("grid_height", self.grid_height)
        
        # Reset alive status
        self.alive = True
        
        # Start new episode
        self.start_episode()
    
    def start_episode(self):
        """Start a new training episode"""
        self.episode_count += 1
        self.stats.episode = self.episode_count
        self.stats.total_reward = 0.0
        
        # Ensure snake is alive at episode start
        self.alive = True
        self.stats.episode_length = 0
        self.stats.foods_eaten = 0
        
        self.current_state = None
        self.last_action = None
        self.game_active = True
        
        # Reset intermediate reward tracking
        self.reset_intermediate_reward_tracking()
        
        print(f"🎬 Starting episode {self.episode_count} for {self.snake_id}")
    
    def end_episode(self):
        """End the current training episode"""
        self.episode_rewards.append(self.stats.total_reward)
        self.episode_lengths.append(self.stats.episode_length)
        
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        print(f"🏁 Episode {self.episode_count} ended for {self.snake_id}")
        print(f"   Reward: {self.stats.total_reward:.2f} (avg: {avg_reward:.2f})")
        print(f"   Length: {self.stats.episode_length} (avg: {avg_length:.1f})")
        print(f"   Foods: {self.stats.foods_eaten}, Epsilon: {self.epsilon:.3f}")
        
        self.game_active = False
    
    def get_public_state(self) -> Dict[str, Any]:
        """Get public state for monitoring with HRM metrics"""
        base_state = {
            "snake_id": self.snake_id,
            "episode": self.episode_count,
            "epsilon": self.epsilon,
            "total_reward": self.stats.total_reward,
            "foods_eaten": self.stats.foods_eaten,
            "deaths": self.stats.deaths,
            "memory_size": len(self.memory),
            "training_steps": self.training_step,
            "game_active": self.game_active,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        # Add network type with HRM indication
        if TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE and HRM_AVAILABLE and self.config.use_hrm:
            base_state["network_type"] = "HRM-Enhanced DQN"
            
            # Add HRM-specific metrics
            if hasattr(self, 'hierarchical_rewards'):
                base_state["avg_hierarchical_reward"] = (
                    np.mean(self.hierarchical_rewards) if self.hierarchical_rewards else 0
                )
                base_state["hierarchical_reward_std"] = (
                    np.std(self.hierarchical_rewards) if len(self.hierarchical_rewards) > 1 else 0
                )
            
            # Add HRM system metrics if available
            if hasattr(self, 'q_network') and hasattr(self.q_network, 'get_hrm_metrics'):
                try:
                    hrm_metrics = self.q_network.get_hrm_metrics()
                    base_state.update({
                        "active_goals": hrm_metrics.get('active_goals', 0),
                        "completed_goals": hrm_metrics.get('completed_goals', 0),
                        "total_goals": hrm_metrics.get('total_goals', 0),
                        "avg_option_success_rate": hrm_metrics.get('average_option_success_rate', 0),
                        "hrm_step_count": hrm_metrics.get('step_count', 0)
                    })
                except Exception as e:
                    print(f"⚠️ Error getting HRM metrics: {e}")
                    
        elif TORCH_AVAILABLE and ENHANCED_DQN_AVAILABLE:
            base_state["network_type"] = "Enhanced DQN"
        elif TORCH_AVAILABLE:
            base_state["network_type"] = "PyTorch DQN"
        else:
            base_state["network_type"] = "Simple"
        
        return base_state

if __name__ == "__main__":
    # Test the AI training agent
    from multi_agent_framework import AgentRegistry
    
    config = TrainingConfig(
        hidden_size=128,
        learning_rate=0.001,
        epsilon_decay=0.99,
        memory_size=5000
    )
    
    registry = AgentRegistry()
    
    # Create AI training agents for both snakes
    ai_agent_a = AITrainingAgent("A", config)
    ai_agent_b = AITrainingAgent("B", config)
    
    registry.register_agent(ai_agent_a)
    registry.register_agent(ai_agent_b)
    
    registry.start_all_agents()
    
    try:
        time.sleep(5)
        
        # Simulate some game events
        print("\n🎮 Testing AI agents...")
        
        # Simulate game start
        for agent in [ai_agent_a, ai_agent_b]:
            agent.send_message(MessageType.RESET_GAME, agent.agent_id, {
                "config": {"grid_width": 15, "grid_height": 15}
            })
        
        time.sleep(2)
        
        # Check states
        print(f"\n📊 AI Agent A state: {ai_agent_a.get_public_state()}")
        print(f"📊 AI Agent B state: {ai_agent_b.get_public_state()}")
        
    finally:
        registry.stop_all_agents()