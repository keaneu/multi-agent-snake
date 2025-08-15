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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, using simplified Q-learning")

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
    
    # Training parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    
    # Training schedule
    train_freq: int = 4  # Train every N steps
    save_freq: int = 1000  # Save model every N episodes
    
    # Reward system
    food_reward: float = 10.0
    death_penalty: float = -10.0
    step_penalty: float = -0.01
    survival_reward: float = 0.1
    
    # Model persistence
    model_save_path: str = "snake_ai_model.pth"
    stats_save_path: str = "training_stats.json"

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
        self.foods = []
        self.obstacles = []
        self.other_snakes = {}
        
        # Training components
        self.state_size = self.calculate_state_size()
        self.action_size = len(Action)
        
        # Initialize neural network
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQN(self.state_size, self.config.hidden_size, self.action_size).to(self.device)
            self.target_network = DQN(self.state_size, self.config.hidden_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
            self.update_target_network()
        else:
            self.q_network = SimpleDQN(self.state_size, self.config.hidden_size, self.action_size, self.config.learning_rate)
            self.target_network = SimpleDQN(self.state_size, self.config.hidden_size, self.action_size, self.config.learning_rate)
        
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
    
    def choose_action(self, state: np.ndarray) -> Action:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
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
        """Calculate reward based on game events"""
        reward = 0.0
        
        if event_type == "food_eaten":
            reward = self.config.food_reward
        elif event_type == "collision":
            reward = self.config.death_penalty
        elif event_type == "step":
            reward = self.config.step_penalty
        elif event_type == "survival":
            reward = self.config.survival_reward
        
        return reward
    
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
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Copy weights for simple network
            self.target_network.W1 = self.q_network.W1.copy()
            self.target_network.b1 = self.q_network.b1.copy()
            self.target_network.W2 = self.q_network.W2.copy()
            self.target_network.b2 = self.q_network.b2.copy()
            self.target_network.W3 = self.q_network.W3.copy()
            self.target_network.b3 = self.q_network.b3.copy()
    
    def save_model(self):
        """Save the trained model"""
        try:
            if TORCH_AVAILABLE:
                torch.save({
                    'model_state_dict': self.q_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'episode': self.episode_count,
                    'stats': self.stats
                }, self.config.model_save_path)
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
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        
        # Start new episode
        self.start_episode()
    
    def start_episode(self):
        """Start a new training episode"""
        self.episode_count += 1
        self.stats.episode = self.episode_count
        self.stats.total_reward = 0.0
        self.stats.episode_length = 0
        self.stats.foods_eaten = 0
        
        self.current_state = None
        self.last_action = None
        self.game_active = True
        
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
        """Get public state for monitoring"""
        return {
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
            "network_type": "PyTorch" if TORCH_AVAILABLE else "Simple"
        }

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