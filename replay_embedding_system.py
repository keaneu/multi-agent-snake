"""
Replay Embedding System - Extracts strategic patterns from high-score replays
Feeds champion gameplay patterns into RNN embeddings for enhanced AI training
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class ReplayPattern:
    """Extracted strategic pattern from a high-score replay"""
    state_sequence: np.ndarray  # Sequence of game states
    action_sequence: np.ndarray  # Sequence of actions taken
    reward_sequence: np.ndarray  # Sequence of rewards received
    final_score: int
    pattern_type: str  # 'food_acquisition', 'collision_avoidance', 'strategic_navigation'

class ReplayFeatureExtractor:
    """Extracts strategic features from replay JSON files"""
    
    def __init__(self):
        self.feature_dim = 33  # Match current state representation
        
    def load_replay(self, replay_path: str) -> Dict:
        """Load replay data from JSON file"""
        try:
            with open(replay_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading replay {replay_path}: {e}")
            return None
    
    def extract_state_features(self, frame: Dict) -> np.ndarray:
        """Extract 33-dimensional state features from a replay frame"""
        features = np.zeros(self.feature_dim)
        
        # Snake positions (normalized)
        snakes = frame.get('snakes', {})
        for snake_id in ['A', 'B']:
            if snake_id in snakes and snakes[snake_id]:
                head_x, head_y = snakes[snake_id][0]
                # Head position (2 features)
                idx = 0 if snake_id == 'A' else 2
                features[idx] = head_x / 20.0
                features[idx + 1] = head_y / 20.0
                
                # Snake length (1 feature)
                length_idx = 4 if snake_id == 'A' else 5
                features[length_idx] = min(len(snakes[snake_id]) / 10.0, 1.0)
        
        # Food positions (up to 3 foods, 6 features)
        foods = frame.get('food', [])
        for i, food in enumerate(foods[:3]):
            if food:
                x, y = food
                features[6 + i*2] = x / 20.0
                features[6 + i*2 + 1] = y / 20.0
        
        # Obstacle positions (up to 4 obstacles, 8 features)
        obstacles = frame.get('obstacles', [])
        for i, obstacle in enumerate(obstacles[:4]):
            if obstacle:
                x, y = obstacle
                features[12 + i*2] = x / 20.0
                features[12 + i*2 + 1] = y / 20.0
        
        # Scores (2 features)
        scores = frame.get('scores', {})
        features[20] = scores.get('A', 0) / 100.0  # Normalized score
        features[21] = scores.get('B', 0) / 100.0
        
        # Distance features (remaining features)
        if 'A' in snakes and snakes['A'] and foods:
            head_x, head_y = snakes['A'][0]
            # Distance to nearest food
            min_food_dist = min([abs(head_x - fx) + abs(head_y - fy) for fx, fy in foods])
            features[22] = min_food_dist / 40.0  # Max possible distance on 20x20 grid
            
            # Distance to walls
            features[23] = head_x / 20.0  # Distance to left wall
            features[24] = (20 - head_x) / 20.0  # Distance to right wall
            features[25] = head_y / 20.0  # Distance to top wall
            features[26] = (20 - head_y) / 20.0  # Distance to bottom wall
        
        # Additional strategic features
        features[27] = len(foods) / 3.0  # Food density
        features[28] = len(obstacles) / 4.0  # Obstacle density
        
        # Game phase encoding
        phase = frame.get('game_phase', 'running')
        features[29] = 1.0 if phase == 'running' else 0.0
        
        # Remaining features for future use
        features[30:33] = 0.0
        
        return features
    
    def classify_pattern_type(self, frames: List[Dict], final_score: int) -> str:
        """Classify the type of strategic pattern demonstrated"""
        if not frames:
            return 'unknown'
        
        food_eaten_count = 0
        collision_avoidances = 0
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Count food eaten
            prev_foods = len(prev_frame.get('food', []))
            curr_foods = len(curr_frame.get('food', []))
            if curr_foods < prev_foods:
                food_eaten_count += 1
            
            # Count near-collision avoidances (simplified heuristic)
            snakes = curr_frame.get('snakes', {})
            for snake_id in ['A', 'B']:
                if snake_id in snakes and snakes[snake_id]:
                    head_x, head_y = snakes[snake_id][0]
                    # Check if near walls or obstacles
                    if head_x <= 1 or head_x >= 18 or head_y <= 1 or head_y >= 18:
                        collision_avoidances += 1
        
        # Classify based on dominant behavior
        if food_eaten_count >= 3:
            return 'food_acquisition'
        elif collision_avoidances >= len(frames) * 0.3:
            return 'collision_avoidance'
        else:
            return 'strategic_navigation'
    
    def extract_replay_patterns(self, replay_path: str) -> Optional[ReplayPattern]:
        """Extract strategic patterns from a single replay file"""
        replay_data = self.load_replay(replay_path)
        if not replay_data:
            return None
        
        frames = replay_data.get('frames', [])
        if len(frames) < 5:  # Too short to be meaningful
            return None
        
        # Extract state sequence
        state_sequence = []
        for frame in frames:
            state_features = self.extract_state_features(frame)
            state_sequence.append(state_features)
        
        # Create action sequence (simplified - based on movement patterns)
        action_sequence = self.infer_actions_from_movement(frames)
        
        # Create reward sequence (based on score changes)
        reward_sequence = self.calculate_reward_sequence(frames)
        
        # Get final score and classify pattern
        final_score = replay_data.get('final_score', 0)
        pattern_type = self.classify_pattern_type(frames, final_score)
        
        return ReplayPattern(
            state_sequence=np.array(state_sequence),
            action_sequence=np.array(action_sequence),
            reward_sequence=np.array(reward_sequence),
            final_score=final_score,
            pattern_type=pattern_type
        )
    
    def infer_actions_from_movement(self, frames: List[Dict]) -> List[int]:
        """Infer action sequence from snake movement patterns"""
        actions = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Get snake A movement (primary focus)
            prev_snakes = prev_frame.get('snakes', {})
            curr_snakes = curr_frame.get('snakes', {})
            
            action = 0  # Default: straight
            
            if 'A' in prev_snakes and 'A' in curr_snakes:
                if prev_snakes['A'] and curr_snakes['A']:
                    prev_head = prev_snakes['A'][0]
                    curr_head = curr_snakes['A'][0]
                    
                    # Calculate movement direction
                    dx = curr_head[0] - prev_head[0]
                    dy = curr_head[1] - prev_head[1]
                    
                    # Simplified action inference
                    if abs(dx) > abs(dy):
                        action = 1 if dx > 0 else 2  # Left/Right relative to movement
                    else:
                        action = 1 if dy > 0 else 2  # Up/Down relative to movement
            
            actions.append(action)
        
        return actions
    
    def calculate_reward_sequence(self, frames: List[Dict]) -> List[float]:
        """Calculate reward sequence based on score changes and game events"""
        rewards = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            reward = 0.0
            
            # Score-based rewards
            prev_scores = prev_frame.get('scores', {})
            curr_scores = curr_frame.get('scores', {})
            
            for snake_id in ['A', 'B']:
                prev_score = prev_scores.get(snake_id, 0)
                curr_score = curr_scores.get(snake_id, 0)
                score_diff = curr_score - prev_score
                
                if score_diff > 0:
                    reward += score_diff * 0.1  # Positive reward for score increase
            
            # Food consumption detection
            prev_foods = len(prev_frame.get('food', []))
            curr_foods = len(curr_frame.get('food', []))
            if curr_foods < prev_foods:
                reward += 1.0  # Reward for eating food
            
            # Survival reward
            reward += 0.01  # Small reward for surviving each frame
            
            rewards.append(reward)
        
        return rewards

class ReplayRNNEmbedding(nn.Module):
    """RNN-based embedding layer for high-score replay patterns"""
    
    def __init__(self, input_size: int = 33, hidden_size: int = 64, embedding_size: int = 32, num_layers: int = 2):
        super(ReplayRNNEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        
        # LSTM for processing replay sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Attention mechanism for focusing on important patterns
        self.attention = nn.Linear(hidden_size, 1)
        
        # Pattern type embedding
        self.pattern_embedding = nn.Embedding(4, 16)  # 4 pattern types
        
        # Final embedding layer
        self.embedding_layer = nn.Linear(hidden_size + 16, embedding_size)
        
    def forward(self, replay_sequences: torch.Tensor, pattern_types: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for replay embedding
        
        Args:
            replay_sequences: (batch_size, seq_len, input_size)
            pattern_types: (batch_size,) - pattern type indices
        
        Returns:
            embeddings: (batch_size, embedding_size)
        """
        batch_size, seq_len, _ = replay_sequences.shape
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(replay_sequences)
        
        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Get pattern type embeddings
        pattern_emb = self.pattern_embedding(pattern_types)
        
        # Combine context and pattern embeddings
        combined = torch.cat([context_vector, pattern_emb], dim=1)
        
        # Final embedding
        embedding = torch.tanh(self.embedding_layer(combined))
        
        return embedding

class HighScoreReplayManager:
    """Manages high-score replay data for AI training enhancement"""
    
    def __init__(self, replay_dir: str = "replays", min_score_threshold: int = 10):
        self.replay_dir = replay_dir
        self.min_score_threshold = min_score_threshold
        self.extractor = ReplayFeatureExtractor()
        self.replay_patterns = []
        self.pattern_type_map = {
            'food_acquisition': 0,
            'collision_avoidance': 1,
            'strategic_navigation': 2,
            'unknown': 3
        }
        
        # Load existing patterns
        self.load_high_score_patterns()
    
    def load_high_score_patterns(self):
        """Load and process all high-score replay files"""
        if not os.path.exists(self.replay_dir):
            print(f"⚠️  Replay directory {self.replay_dir} not found")
            return
        
        replay_files = [f for f in os.listdir(self.replay_dir) if f.endswith('.json')]
        high_score_files = []
        
        # Filter for high-score replays
        for filename in replay_files:
            try:
                # Extract score from filename (e.g., episode_123_score_24.json)
                if '_score_' in filename:
                    score_part = filename.split('_score_')[1].split('.')[0]
                    score = int(score_part)
                    if score >= self.min_score_threshold:
                        high_score_files.append((filename, score))
            except (ValueError, IndexError):
                continue
        
        # Sort by score (highest first)
        high_score_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 Loading {len(high_score_files)} high-score replays (≥{self.min_score_threshold} points)")
        
        # Process each high-score replay
        for filename, score in high_score_files:
            replay_path = os.path.join(self.replay_dir, filename)
            pattern = self.extractor.extract_replay_patterns(replay_path)
            
            if pattern:
                self.replay_patterns.append(pattern)
                print(f"✅ Loaded {filename}: {score} points, {pattern.pattern_type}, {len(pattern.state_sequence)} frames")
        
        print(f"🏆 Total patterns loaded: {len(self.replay_patterns)}")
    
    def get_replay_embeddings(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get batched replay embeddings for training"""
        if not self.replay_patterns or not TORCH_AVAILABLE:
            return None, None
        
        # Sample patterns for batch
        sampled_patterns = np.random.choice(self.replay_patterns, min(batch_size, len(self.replay_patterns)), replace=False)
        
        # Prepare sequences (pad to same length)
        max_len = max(len(pattern.state_sequence) for pattern in sampled_patterns)
        sequences = []
        pattern_types = []
        
        for pattern in sampled_patterns:
            # Pad sequence to max length
            seq = pattern.state_sequence
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, padding])
            
            sequences.append(seq)
            pattern_types.append(self.pattern_type_map.get(pattern.pattern_type, 3))
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(np.array(sequences))
        pattern_types_tensor = torch.LongTensor(pattern_types)
        
        return sequences_tensor, pattern_types_tensor
    
    def get_best_pattern_features(self) -> Optional[np.ndarray]:
        """Get feature vector from the highest-scoring pattern"""
        if not self.replay_patterns:
            return None
        
        # Find highest scoring pattern
        best_pattern = max(self.replay_patterns, key=lambda x: x.final_score)
        
        # Return average state features from best pattern
        return np.mean(best_pattern.state_sequence, axis=0)
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about loaded patterns"""
        if not self.replay_patterns:
            return {}
        
        stats = {
            'total_patterns': len(self.replay_patterns),
            'average_score': np.mean([p.final_score for p in self.replay_patterns]),
            'max_score': max([p.final_score for p in self.replay_patterns]),
            'pattern_types': {},
            'average_length': np.mean([len(p.state_sequence) for p in self.replay_patterns])
        }
        
        # Count pattern types
        for pattern in self.replay_patterns:
            ptype = pattern.pattern_type
            stats['pattern_types'][ptype] = stats['pattern_types'].get(ptype, 0) + 1
        
        return stats

if __name__ == "__main__":
    # Test the replay embedding system
    print("🎮 Testing Replay Embedding System")
    
    manager = HighScoreReplayManager()
    stats = manager.get_pattern_statistics()
    
    print(f"📊 Pattern Statistics: {stats}")
    
    if TORCH_AVAILABLE and manager.replay_patterns:
        # Test RNN embedding
        embedding_model = ReplayRNNEmbedding()
        sequences, pattern_types = manager.get_replay_embeddings(batch_size=4)
        
        if sequences is not None:
            print(f"🧠 Testing RNN embedding with batch shape: {sequences.shape}")
            embeddings = embedding_model(sequences, pattern_types)
            print(f"✅ Generated embeddings shape: {embeddings.shape}")
            print(f"🎯 Sample embedding: {embeddings[0][:8].detach().numpy()}")