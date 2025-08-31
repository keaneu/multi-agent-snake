import numpy as np
import random
import pickle
import os
from collections import deque

class SnakeEnvironment:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        # Initialize snake in center
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (1, 0)  # Right
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self.get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake:
                return food
    
    def step(self, action):
        # Actions: 0=straight, 1=left, 2=right
        self.steps += 1
        
        # Convert action to direction change
        if action == 1:  # Turn left
            if self.direction == (1, 0): self.direction = (0, -1)
            elif self.direction == (0, -1): self.direction = (-1, 0)
            elif self.direction == (-1, 0): self.direction = (0, 1)
            elif self.direction == (0, 1): self.direction = (1, 0)
        elif action == 2:  # Turn right
            if self.direction == (1, 0): self.direction = (0, 1)
            elif self.direction == (0, 1): self.direction = (-1, 0)
            elif self.direction == (-1, 0): self.direction = (0, -1)
            elif self.direction == (0, -1): self.direction = (1, 0)
        
        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check collisions
        done = False
        reward = 0
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            done = True
            reward = -10
        # Self collision
        elif new_head in self.snake:
            done = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            
            # Check if food eaten
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self._place_food()
            else:
                self.snake.pop()
                reward = -0.1  # Small negative reward for each step
        
        # Timeout penalty
        if self.steps > 1000:
            done = True
            reward = -5
        
        return self.get_state(), reward, done
    
    def get_state(self):
        head = self.snake[0]
        
        # Danger detection (normalized to 0-1)
        danger_straight = self._is_collision(head, self.direction)
        danger_left = self._is_collision(head, self._get_left_direction())
        danger_right = self._is_collision(head, self._get_right_direction())
        
        # Direction (one-hot encoded)
        dir_up = self.direction == (0, -1)
        dir_down = self.direction == (0, 1)
        dir_left = self.direction == (-1, 0)
        dir_right = self.direction == (1, 0)
        
        # Food direction (relative to head)
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        
        # Distance to food (normalized)
        food_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        food_dist_norm = food_dist / (self.width + self.height)
        
        state = [
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right,
            food_dist_norm
        ]
        
        return tuple(state)
    
    def _is_collision(self, head, direction):
        new_pos = (head[0] + direction[0], head[1] + direction[1])
        return (new_pos[0] < 0 or new_pos[0] >= self.width or 
                new_pos[1] < 0 or new_pos[1] >= self.height or 
                new_pos in self.snake)
    
    def _get_left_direction(self):
        if self.direction == (1, 0): return (0, -1)
        elif self.direction == (0, -1): return (-1, 0)
        elif self.direction == (-1, 0): return (0, 1)
        elif self.direction == (0, 1): return (1, 0)
    
    def _get_right_direction(self):
        if self.direction == (1, 0): return (0, 1)
        elif self.direction == (0, 1): return (-1, 0)
        elif self.direction == (-1, 0): return (0, -1)
        elif self.direction == (0, -1): return (1, 0)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table: state -> action values
        self.q_table = {}
        
        # Experience replay - Phase 44 Legendary Memory Enhancement
        self.memory = deque(maxlen=1000000)
    
    def get_action(self, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        
        # Get Q-values for current state
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # 3 actions
        return self.q_table[state]
    
    def update_q_table(self, state, action, reward, next_state, done):
        current_q = self.get_q_values(state)[action]
        
        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.gamma * max(next_q_values)
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
            print(f"Loaded model from {filepath}")

def train_agent(episodes=10000):
    env = SnakeEnvironment()
    agent = QLearningAgent()
    
    # Try to load existing model
    model_path = "snake_q_model.pkl"
    agent.load_model(model_path)
    
    scores = []
    best_score = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update_q_table(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(env.score)
        agent.decay_epsilon()
        
        # Track best performance
        if env.score > best_score:
            best_score = env.score
            agent.save_model(model_path)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Best: {best_score}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores

def test_agent(agent, num_games=10):
    env = SnakeEnvironment()
    scores = []
    
    # Set epsilon to 0 for testing (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for game in range(num_games):
        state = env.reset()
        
        while True:
            action = agent.get_action(state)
            state, _, done = env.step(action)
            
            if done:
                break
        
        scores.append(env.score)
        print(f"Game {game + 1}: Score = {env.score}")
    
    agent.epsilon = original_epsilon
    print(f"Average score over {num_games} games: {np.mean(scores):.2f}")
    return scores

if __name__ == "__main__":
    print("Training Snake RL Agent...")
    agent, training_scores = train_agent(episodes=5000)
    
    print("\nTesting trained agent...")
    test_scores = test_agent(agent, num_games=5)
    
    print(f"\nTraining complete!")
    print(f"Final average score: {np.mean(training_scores[-100:]):.2f}")
    print(f"States learned: {len(agent.q_table)}")