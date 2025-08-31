"""
Intermediate Replay Manager for 15-30 Score Range
Specialized replay loading and processing for intermediate skill development
"""

import numpy as np
import json
import os
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class IntermediateReplayPattern:
    """Replay pattern specifically for intermediate score range (15-30)"""
    state_sequence: np.ndarray
    action_sequence: np.ndarray
    score_progression: np.ndarray
    final_score: int
    pattern_type: str
    skill_level: str  # 'breakthrough', 'consolidation', 'mastery'

class IntermediateReplayManager:
    """
    Specialized replay manager for 15-30 score range
    Loads replays from replays folder and categorizes by skill progression
    """
    
    def __init__(self, replay_dir: str = "replays", score_range: Tuple[int, int] = (10, 25)):
        self.replay_dir = replay_dir
        self.score_range = score_range
        self.min_score, self.max_score = score_range
        
        # Replay storage by skill level
        self.skill_categories = {
            'breakthrough': [],      # 15-20 range
            'consolidation': [],     # 21-25 range  
            'mastery': []           # 26-30 range
        }
        
        # Pattern analysis
        self.score_distribution = defaultdict(int)
        self.pattern_types = defaultdict(int)
        self.loaded_replays = []
        
        # Load and categorize replays
        self.load_intermediate_replays()
        self.analyze_replay_patterns()
        
        total_patterns = sum(len(patterns) for patterns in self.skill_categories.values())
        print(f"🎯 IntermediateReplayManager initialized")
        print(f"   Score range: {score_range}")
        print(f"   Total patterns: {total_patterns}")
        print(f"   Distribution: {dict(self.skill_categories)}")
    
    def load_intermediate_replays(self):
        """Load replay files in the intermediate score range"""
        if not os.path.exists(self.replay_dir):
            print(f"⚠️ Replay directory not found: {self.replay_dir}")
            return
        
        replay_files_by_score = []
        
        # Find all replay files in target range
        for filename in os.listdir(self.replay_dir):
            if filename.endswith('.json') and 'score_' in filename:
                try:
                    score_part = filename.split('score_')[1].split('.')[0]
                    score = int(score_part)
                    
                    if self.min_score <= score <= self.max_score:
                        filepath = os.path.join(self.replay_dir, filename)
                        replay_files_by_score.append((filepath, score))
                        self.score_distribution[score] += 1
                        
                except (ValueError, IndexError):
                    continue
        
        # Sort by score for systematic loading
        replay_files_by_score.sort(key=lambda x: x[1])
        
        # Load replay data with balanced sampling
        total_loaded = 0
        max_per_score = 3  # Maximum replays per score value
        
        for filepath, score in replay_files_by_score:
            if total_loaded >= 150:  # Limit total replays for memory
                break
                
            try:
                with open(filepath, 'r') as f:
                    replay_data = json.load(f)
                    
                if self._validate_replay_data(replay_data):
                    replay_data['filepath'] = filepath
                    replay_data['final_score'] = score
                    self.loaded_replays.append(replay_data)
                    total_loaded += 1
                    
            except Exception as e:
                print(f"⚠️ Error loading {filepath}: {e}")
        
        print(f"📈 Loaded {len(self.loaded_replays)} intermediate replays")
    
    def _validate_replay_data(self, replay_data: Dict) -> bool:
        """Validate replay data structure"""
        required_fields = ['frames', 'final_score']
        
        for field in required_fields:
            if field not in replay_data:
                return False
        
        frames = replay_data.get('frames', [])
        if len(frames) < 15:  # Minimum meaningful game length
            return False
        
        # Check frame structure
        for frame in frames[:3]:
            if not all(key in frame for key in ['snakes', 'food']):
                return False
        
        return True
    
    def analyze_replay_patterns(self):
        """Analyze and categorize replay patterns by skill level"""
        for replay_data in self.loaded_replays:
            try:
                patterns = self._extract_replay_patterns(replay_data)
                skill_level = self._categorize_skill_level(replay_data['final_score'])
                
                for pattern in patterns:
                    pattern.skill_level = skill_level
                    self.skill_categories[skill_level].append(pattern)
                    self.pattern_types[pattern.pattern_type] += 1
                    
            except Exception as e:
                print(f"⚠️ Error analyzing replay: {e}")
        
        # Print analysis summary
        print(f"\n📊 Replay Pattern Analysis:")
        for skill, patterns in self.skill_categories.items():
            print(f"   {skill}: {len(patterns)} patterns")
        print(f"   Pattern types: {dict(self.pattern_types)}")
    
    def _categorize_skill_level(self, final_score: int) -> str:
        """Categorize score into skill levels"""
        if 10 <= final_score <= 15:
            return 'breakthrough'
        elif 16 <= final_score <= 20:
            return 'consolidation'
        elif 21 <= final_score <= 25:
            return 'mastery'
        else:
            return 'breakthrough'  # Default fallback
    
    def _extract_replay_patterns(self, replay_data: Dict) -> List[IntermediateReplayPattern]:
        """Extract patterns from replay data"""
        frames = replay_data['frames']
        final_score = replay_data['final_score']
        patterns = []
        
        # Extract sequential patterns
        pattern_length = 15  # 15-frame sequences
        step_size = 5       # Overlap sequences for more data
        
        for start_idx in range(0, len(frames) - pattern_length, step_size):
            end_idx = start_idx + pattern_length
            frame_sequence = frames[start_idx:end_idx]
            
            # Extract state and action sequences
            state_sequence = []
            action_sequence = []
            score_progression = []
            
            prev_snake_length = 3  # Initial snake length
            
            for i, frame in enumerate(frame_sequence):
                # Extract state features
                state_vector = self._frame_to_state_vector(frame)
                if state_vector is not None:
                    state_sequence.append(state_vector)
                    
                    # Estimate action (simplified)
                    action = self._estimate_action_from_frames(frame_sequence, i)
                    action_sequence.append(action)
                    
                    # Track score progression
                    current_length = self._get_snake_length_from_frame(frame)
                    score = max(0, current_length - 3)
                    score_progression.append(score)
            
            if len(state_sequence) == pattern_length:
                # Determine pattern type
                pattern_type = self._classify_pattern_type(frame_sequence, score_progression)
                
                pattern = IntermediateReplayPattern(
                    state_sequence=np.array(state_sequence),
                    action_sequence=np.array(action_sequence),
                    score_progression=np.array(score_progression),
                    final_score=final_score,
                    pattern_type=pattern_type,
                    skill_level=""  # Will be set by caller
                )
                patterns.append(pattern)
        
        return patterns
    
    def _frame_to_state_vector(self, frame: Dict) -> Optional[np.ndarray]:
        """Convert frame to state vector compatible with training system"""
        try:
            # Extract snake data (assuming snake A is the target)
            snakes = frame.get('snakes', {})
            if 'A' not in snakes or not snakes['A']:
                return None
            
            snake_segments = snakes['A']
            head_pos = snake_segments[0] if snake_segments else [10, 10]
            
            # Extract food and obstacles
            foods = frame.get('food', [])
            obstacles = frame.get('obstacles', [])
            
            # Build state vector (33 dimensions to match training system)
            state_features = []
            
            # 1. Head position (normalized)
            state_features.extend([head_pos[0] / 20.0, head_pos[1] / 20.0])
            
            # 2. Snake length (normalized)
            state_features.append(len(snake_segments) / 50.0)
            
            # 3. Direction encoding (simplified)
            if len(snake_segments) >= 2:
                direction_vector = [
                    snake_segments[0][0] - snake_segments[1][0],
                    snake_segments[0][1] - snake_segments[1][1]
                ]
                # Convert to one-hot-like encoding
                if direction_vector == [0, -1]:      # UP
                    state_features.extend([1, 0, 0, 0])
                elif direction_vector == [0, 1]:     # DOWN
                    state_features.extend([0, 1, 0, 0])
                elif direction_vector == [-1, 0]:    # LEFT
                    state_features.extend([0, 0, 1, 0])
                elif direction_vector == [1, 0]:     # RIGHT
                    state_features.extend([0, 0, 0, 1])
                else:
                    state_features.extend([0, 0, 0, 0])
            else:
                state_features.extend([0, 0, 0, 1])  # Default RIGHT
            
            # 4. Food information (closest 3 foods)
            food_features = self._get_food_features(head_pos, foods)
            state_features.extend(food_features)
            
            # 5. Danger detection
            danger_features = self._get_danger_features(head_pos, snake_segments, obstacles)
            state_features.extend(danger_features)
            
            # 6. Wall distances
            wall_features = self._get_wall_features(head_pos)
            state_features.extend(wall_features)
            
            # Pad or truncate to exactly 33 features
            while len(state_features) < 33:
                state_features.append(0.0)
            
            return np.array(state_features[:33], dtype=np.float32)
            
        except Exception as e:
            print(f"⚠️ Error converting frame to state: {e}")
            return None
    
    def _get_food_features(self, head_pos: List[int], foods: List[List[int]]) -> List[float]:
        """Get food-related features (12 features for 3 closest foods)"""
        features = []
        
        if not foods:
            return [0.0] * 12
        
        # Calculate distances and sort
        food_distances = []
        for food in foods:
            dist = abs(food[0] - head_pos[0]) + abs(food[1] - head_pos[1])
            food_distances.append((dist, food))
        
        food_distances.sort()
        
        # Features for closest 3 foods
        for i in range(min(3, len(food_distances))):
            dist, food = food_distances[i]
            
            # Normalized distance
            features.append(min(dist / 40.0, 1.0))
            
            # Direction (normalized)
            dx = (food[0] - head_pos[0]) / 20.0
            dy = (food[1] - head_pos[1]) / 20.0
            features.extend([dx, dy])
            
            # Quadrant indicator
            quadrant = 0
            if dx > 0 and dy > 0: quadrant = 0.25
            elif dx < 0 and dy > 0: quadrant = 0.50
            elif dx < 0 and dy < 0: quadrant = 0.75
            elif dx > 0 and dy < 0: quadrant = 1.0
            features.append(quadrant)
        
        # Pad if fewer than 3 foods
        while len(features) < 12:
            features.append(0.0)
        
        return features
    
    def _get_danger_features(self, head_pos: List[int], snake_segments: List[List[int]], 
                           obstacles: List[List[int]]) -> List[float]:
        """Get danger detection features (8 features)"""
        features = []
        
        # Check 8 directions around head
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            check_pos = [head_pos[0] + dx, head_pos[1] + dy]
            danger = 0.0
            
            # Check wall collision
            if check_pos[0] < 0 or check_pos[0] >= 20 or check_pos[1] < 0 or check_pos[1] >= 20:
                danger = 1.0
            # Check snake body collision
            elif check_pos in snake_segments:
                danger = 1.0
            # Check obstacle collision
            elif check_pos in obstacles:
                danger = 1.0
            
            features.append(danger)
        
        return features
    
    def _get_wall_features(self, head_pos: List[int]) -> List[float]:
        """Get wall distance features (4 features)"""
        x, y = head_pos
        
        # Normalized distances to walls
        up_dist = y / 20.0
        down_dist = (19 - y) / 20.0
        left_dist = x / 20.0
        right_dist = (19 - x) / 20.0
        
        return [up_dist, down_dist, left_dist, right_dist]
    
    def _get_snake_length_from_frame(self, frame: Dict) -> int:
        """Get snake length from frame"""
        snakes = frame.get('snakes', {})
        if 'A' in snakes and snakes['A']:
            return len(snakes['A'])
        return 3
    
    def _estimate_action_from_frames(self, frame_sequence: List[Dict], current_idx: int) -> int:
        """Estimate action taken based on movement between frames"""
        if current_idx >= len(frame_sequence) - 1:
            return 0  # STRAIGHT
        
        current_frame = frame_sequence[current_idx]
        next_frame = frame_sequence[current_idx + 1]
        
        # Get snake head positions
        current_snake = current_frame.get('snakes', {}).get('A', [])
        next_snake = next_frame.get('snakes', {}).get('A', [])
        
        if not current_snake or not next_snake:
            return 0  # STRAIGHT
        
        current_head = current_snake[0]
        next_head = next_snake[0]
        
        # Calculate movement vector
        dx = next_head[0] - current_head[0]
        dy = next_head[1] - current_head[1]
        
        # Get current direction (if snake has body)
        if len(current_snake) >= 2:
            current_dir = [
                current_snake[0][0] - current_snake[1][0],
                current_snake[0][1] - current_snake[1][1]
            ]
        else:
            current_dir = [1, 0]  # Default RIGHT
        
        # Determine action based on direction change
        if [dx, dy] == current_dir:
            return 0  # STRAIGHT
        elif self._is_left_turn(current_dir, [dx, dy]):
            return 1  # LEFT
        elif self._is_right_turn(current_dir, [dx, dy]):
            return 2  # RIGHT
        else:
            return 0  # STRAIGHT (fallback)
    
    def _is_left_turn(self, current_dir: List[int], new_dir: List[int]) -> bool:
        """Check if movement is a left turn"""
        left_mappings = {
            (0, -1): (-1, 0),  # UP -> LEFT
            (-1, 0): (0, 1),   # LEFT -> DOWN
            (0, 1): (1, 0),    # DOWN -> RIGHT
            (1, 0): (0, -1)    # RIGHT -> UP
        }
        return left_mappings.get(tuple(current_dir)) == tuple(new_dir)
    
    def _is_right_turn(self, current_dir: List[int], new_dir: List[int]) -> bool:
        """Check if movement is a right turn"""
        right_mappings = {
            (0, -1): (1, 0),   # UP -> RIGHT
            (1, 0): (0, 1),    # RIGHT -> DOWN
            (0, 1): (-1, 0),   # DOWN -> LEFT
            (-1, 0): (0, -1)   # LEFT -> UP
        }
        return right_mappings.get(tuple(current_dir)) == tuple(new_dir)
    
    def _classify_pattern_type(self, frame_sequence: List[Dict], score_progression: np.ndarray) -> str:
        """Classify the type of pattern in the sequence"""
        # Analyze score progression
        score_change = score_progression[-1] - score_progression[0]
        
        # Analyze food acquisition
        food_acquired = score_change > 0
        
        # Analyze movement characteristics
        if food_acquired and score_change >= 2:
            return "efficient_food_collection"
        elif food_acquired:
            return "food_acquisition"
        elif self._has_collision_avoidance(frame_sequence):
            return "collision_avoidance"
        elif self._has_strategic_movement(frame_sequence):
            return "strategic_navigation"
        else:
            return "survival_pattern"
    
    def _has_collision_avoidance(self, frame_sequence: List[Dict]) -> bool:
        """Check if sequence contains collision avoidance patterns"""
        # Simplified: check if snake navigates near obstacles
        for frame in frame_sequence:
            snakes = frame.get('snakes', {})
            if 'A' in snakes and snakes['A']:
                head_pos = snakes['A'][0]
                obstacles = frame.get('obstacles', [])
                
                # Check proximity to obstacles
                for obstacle in obstacles:
                    dist = abs(obstacle[0] - head_pos[0]) + abs(obstacle[1] - head_pos[1])
                    if dist <= 2:  # Close to obstacle
                        return True
        return False
    
    def _has_strategic_movement(self, frame_sequence: List[Dict]) -> bool:
        """Check if sequence shows strategic movement patterns"""
        # Simplified: check for consistent direction changes
        direction_changes = 0
        
        for i in range(len(frame_sequence) - 1):
            current_frame = frame_sequence[i]
            next_frame = frame_sequence[i + 1]
            
            current_snake = current_frame.get('snakes', {}).get('A', [])
            next_snake = next_frame.get('snakes', {}).get('A', [])
            
            if len(current_snake) >= 2 and len(next_snake) >= 2:
                current_dir = [
                    current_snake[0][0] - current_snake[1][0],
                    current_snake[0][1] - current_snake[1][1]
                ]
                next_dir = [
                    next_snake[0][0] - next_snake[1][0],
                    next_snake[0][1] - next_snake[1][1]
                ]
                
                if current_dir != next_dir:
                    direction_changes += 1
        
        # Strategic movement has moderate direction changes
        return 2 <= direction_changes <= 5
    
    def get_patterns_by_skill_level(self, skill_level: str, max_patterns: int = 10) -> List[IntermediateReplayPattern]:
        """Get patterns for specific skill level"""
        if skill_level not in self.skill_categories:
            return []
        
        patterns = self.skill_categories[skill_level]
        if len(patterns) <= max_patterns:
            return patterns
        else:
            return random.sample(patterns, max_patterns)
    
    def get_mixed_patterns(self, total_patterns: int = 32) -> List[IntermediateReplayPattern]:
        """Get mixed patterns across all skill levels"""
        all_patterns = []
        
        # Balance across skill levels
        per_skill = total_patterns // 3
        
        for skill_level in ['breakthrough', 'consolidation', 'mastery']:
            skill_patterns = self.get_patterns_by_skill_level(skill_level, per_skill)
            all_patterns.extend(skill_patterns)
        
        # Fill remaining slots with random patterns
        while len(all_patterns) < total_patterns:
            all_skill_patterns = []
            for patterns in self.skill_categories.values():
                all_skill_patterns.extend(patterns)
            
            if all_skill_patterns:
                all_patterns.append(random.choice(all_skill_patterns))
            else:
                break
        
        return all_patterns[:total_patterns]
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded replays"""
        total_patterns = sum(len(patterns) for patterns in self.skill_categories.values())
        
        return {
            'total_replays': len(self.loaded_replays),
            'total_patterns': total_patterns,
            'score_range': self.score_range,
            'score_distribution': dict(self.score_distribution),
            'skill_distribution': {k: len(v) for k, v in self.skill_categories.items()},
            'pattern_types': dict(self.pattern_types),
            'average_score': np.mean([r['final_score'] for r in self.loaded_replays]) if self.loaded_replays else 0
        }

if __name__ == "__main__":
    # Test the intermediate replay manager
    print("🎯 Testing Intermediate Replay Manager")
    
    # Create manager
    manager = IntermediateReplayManager(replay_dir="replays", score_range=(10, 25))
    
    # Get statistics
    stats = manager.get_replay_statistics()
    print(f"\n📊 Replay Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test pattern retrieval
    print(f"\n🔍 Testing pattern retrieval...")
    
    # Get breakthrough patterns
    breakthrough_patterns = manager.get_patterns_by_skill_level('breakthrough', 5)
    print(f"   Breakthrough patterns: {len(breakthrough_patterns)}")
    
    # Get mixed patterns
    mixed_patterns = manager.get_mixed_patterns(20)
    print(f"   Mixed patterns: {len(mixed_patterns)}")
    
    if mixed_patterns:
        sample_pattern = mixed_patterns[0]
        print(f"   Sample pattern shape: {sample_pattern.state_sequence.shape}")
        print(f"   Sample pattern type: {sample_pattern.pattern_type}")
        print(f"   Sample skill level: {sample_pattern.skill_level}")
    
    print("✅ Intermediate replay manager test completed!")