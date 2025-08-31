"""
Intermediate Score Training System (15-30 Range)
Specialized training configuration and reward structure for consolidating skills in the 15-30 point range
Uses replays folder data to build upon foundational performance
"""

import numpy as np
import random
import time
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from ai_training_agent import AITrainingAgent, TrainingConfig

@dataclass
class IntermediateTrainingConfig(TrainingConfig):
    """Enhanced configuration for intermediate score range (15-30 points)"""
    
    # Core Training Parameters - Optimized for intermediate skill development
    hidden_size: int = 256
    learning_rate: float = 0.0008  # Slightly reduced for stability
    gamma: float = 0.96  # Enhanced long-term planning
    
    # Exploration Parameters - Reduced for skill consolidation
    epsilon_start: float = 0.4
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.996
    
    # Experience Replay - Enhanced for intermediate learning
    memory_size: int = 15000
    batch_size: int = 48
    target_update_freq: int = 80
    train_freq: int = 3
    
    # INTERMEDIATE SCORE REWARD STRUCTURE (15-30 range focus)
    # Base Rewards - Balanced for consistent intermediate performance
    food_reward: float = 10.0          # Stable food reward for consistency
    death_penalty: float = -6.0        # Moderate penalty to encourage calculated risks
    step_penalty: float = -0.001       # Minimal step penalty for patience
    survival_reward: float = 0.15      # Enhanced survival incentive
    
    # Intermediate Skill Development Rewards
    consistency_reward: float = 2.0           # Reward for consistent 15+ performance
    skill_consolidation_bonus: float = 1.5    # Bonus for demonstrating learned skills
    intermediate_milestone_bonus: float = 3.0  # Bonus for reaching 20, 25, 30 points
    steady_progress_reward: float = 0.8       # Reward for maintaining progress
    
    # Strategic Intermediate Bonuses
    mid_game_strategy_bonus: float = 1.2      # Bonus for good mid-game decisions
    risk_management_bonus: float = 1.0        # Bonus for smart risk assessment
    tactical_patience_bonus: float = 0.6      # Bonus for patient tactical play
    adaptive_movement_bonus: float = 0.4      # Bonus for adaptive movement patterns
    
    # Enhanced Proximity and Safety Rewards
    food_proximity_reward: float = 1.0        # Balanced proximity guidance
    wall_avoidance_reward: float = 0.8        # Safety awareness
    efficient_movement_reward: float = 0.4    # Movement efficiency
    safe_exploration_reward: float = 0.3      # Safe exploration patterns
    
    # Intermediate Performance Tracking
    performance_consistency_threshold: int = 15  # Minimum score for consistency tracking
    skill_demonstration_window: int = 10         # Window for tracking skill demonstration
    
    # Replay System Configuration
    replay_score_range: Tuple[int, int] = (15, 30)  # Target score range for replay learning
    replay_data_path: str = "replays"                # Path to replay data
    min_replay_episodes: int = 20                    # Minimum replay episodes to use
    
    # Model Configuration
    model_save_path: str = "snake_ai_intermediate_15_30.pth"
    stats_save_path: str = "intermediate_training_stats.json"

class IntermediateScoreTrainer:
    """
    Specialized trainer for developing intermediate score range performance (15-30 points)
    Focuses on skill consolidation and consistent performance improvement
    """
    
    def __init__(self, snake_id: str, config: Optional[IntermediateTrainingConfig] = None):
        self.config = config or IntermediateTrainingConfig()
        self.snake_id = snake_id
        
        # Create AI agent with intermediate configuration
        self.ai_agent = AITrainingAgent(snake_id, self.config)
        
        # Intermediate performance tracking
        self.recent_scores = deque(maxlen=20)
        self.consistency_streak = 0
        self.best_intermediate_score = 0
        self.skill_milestones = {
            "first_20": False,
            "first_25": False,
            "first_30": False,
            "consistent_15_plus": False,
            "consistent_20_plus": False
        }
        
        # Replay system for learning from 15-30 score games
        self.replay_episodes = []
        self.load_intermediate_replays()
        
        # Performance metrics
        self.session_stats = {
            "episodes_played": 0,
            "scores_15_plus": 0,
            "scores_20_plus": 0,
            "scores_25_plus": 0,
            "scores_30_plus": 0,
            "consistency_rate": 0.0,
            "avg_score_last_20": 0.0
        }
    
    def load_intermediate_replays(self):
        """Load replay episodes in the 15-30 score range for learning"""
        replay_path = self.config.replay_data_path
        if not os.path.exists(replay_path):
            print(f"⚠️  Replay path {replay_path} not found")
            return
        
        replay_files = []
        target_min, target_max = self.config.replay_score_range
        
        for filename in os.listdir(replay_path):
            if filename.endswith('.json') and 'score_' in filename:
                try:
                    # Extract score from filename
                    score_part = filename.split('score_')[1].split('.')[0]
                    score = int(score_part)
                    
                    if target_min <= score <= target_max:
                        replay_files.append(os.path.join(replay_path, filename))
                except (ValueError, IndexError):
                    continue
        
        # Load replay data
        self.replay_episodes = []
        for filepath in replay_files[:50]:  # Limit to 50 episodes for memory efficiency
            try:
                with open(filepath, 'r') as f:
                    replay_data = json.load(f)
                    self.replay_episodes.append(replay_data)
            except Exception as e:
                print(f"⚠️  Error loading replay {filepath}: {e}")
        
        print(f"📈 Loaded {len(self.replay_episodes)} intermediate replay episodes (scores {target_min}-{target_max})")
    
    def calculate_intermediate_rewards(self, base_reward: float, event_type: str, game_data: Dict[str, Any]) -> float:
        """Calculate enhanced rewards for intermediate score development"""
        enhanced_reward = base_reward
        snake_length = len(self.ai_agent.snake_segments) if self.ai_agent.snake_segments else 3
        
        # 1. Consistency Reward - reward for maintaining 15+ performance
        if event_type == "food_eaten" and snake_length >= 6:  # 15+ score territory
            consistency_bonus = self.config.consistency_reward * min(snake_length / 10, 2.0)
            enhanced_reward += consistency_bonus
        
        # 2. Skill Consolidation Bonus - reward for demonstrating learned behaviors
        if event_type == "survival":
            if hasattr(self.ai_agent, '_recent_positions'):
                # Reward for good movement patterns
                recent_positions = list(self.ai_agent._recent_positions)
                if len(recent_positions) >= 4:
                    unique_positions = len(set(recent_positions))
                    if unique_positions >= 3:  # Good movement variety
                        enhanced_reward += self.config.skill_consolidation_bonus * 0.1
        
        # 3. Intermediate Milestone Bonuses
        if event_type == "food_eaten":
            current_score = (snake_length - 3) * 1  # Approximate score
            milestone_bonus = 0.0
            
            if current_score >= 30 and not self.skill_milestones["first_30"]:
                milestone_bonus = self.config.intermediate_milestone_bonus * 3.0
                self.skill_milestones["first_30"] = True
                print(f"🎯 {self.snake_id} achieved first 30+ score!")
            elif current_score >= 25 and not self.skill_milestones["first_25"]:
                milestone_bonus = self.config.intermediate_milestone_bonus * 2.0
                self.skill_milestones["first_25"] = True
                print(f"🎯 {self.snake_id} achieved first 25+ score!")
            elif current_score >= 20 and not self.skill_milestones["first_20"]:
                milestone_bonus = self.config.intermediate_milestone_bonus * 1.0
                self.skill_milestones["first_20"] = True
                print(f"🎯 {self.snake_id} achieved first 20+ score!")
            
            enhanced_reward += milestone_bonus
        
        # 4. Steady Progress Reward - reward for gradual improvement
        if event_type == "food_eaten" and len(self.recent_scores) >= 5:
            recent_avg = np.mean(list(self.recent_scores)[-5:])
            if recent_avg >= 15:  # Consistent intermediate performance
                progress_bonus = self.config.steady_progress_reward
                enhanced_reward += progress_bonus
        
        # 5. Mid-Game Strategy Bonus - reward for good decisions in 15-25 range
        if event_type == "survival" and 6 <= snake_length <= 10:  # Mid-game range
            if hasattr(self.ai_agent, 'snake_segments') and self.ai_agent.foods:
                head_pos = self.ai_agent.snake_segments[0]
                # Reward for strategic positioning
                food_distances = [abs(f[0] - head_pos[0]) + abs(f[1] - head_pos[1]) for f in self.ai_agent.foods]
                if food_distances and min(food_distances) <= 3:  # Close to food
                    enhanced_reward += self.config.mid_game_strategy_bonus * 0.1
        
        # 6. Risk Management Bonus - reward for avoiding unnecessary risks
        if event_type == "survival":
            dangers = self.ai_agent.get_danger_detection(self.ai_agent.snake_segments[0]) if self.ai_agent.snake_segments else [0, 0, 0]
            danger_count = sum(dangers)
            
            if danger_count <= 1:  # Low danger situation
                enhanced_reward += self.config.risk_management_bonus * 0.05
            elif danger_count >= 2:  # High danger - penalty for poor positioning
                enhanced_reward -= self.config.risk_management_bonus * 0.1
        
        # 7. Tactical Patience Bonus - reward for patient, calculated play
        if event_type == "step" and snake_length >= 5:
            # Track movement efficiency over time
            if not hasattr(self, '_patience_tracker'):
                self._patience_tracker = deque(maxlen=20)
            
            # Simple patience metric: not moving directly away from food
            if hasattr(self.ai_agent, 'snake_segments') and self.ai_agent.foods and self.ai_agent.snake_segments:
                head_pos = self.ai_agent.snake_segments[0]
                nearest_food = min(self.ai_agent.foods, key=lambda f: abs(f[0] - head_pos[0]) + abs(f[1] - head_pos[1]))
                food_distance = abs(nearest_food[0] - head_pos[0]) + abs(nearest_food[1] - head_pos[1])
                
                if hasattr(self, '_prev_food_distance'):
                    if food_distance <= self._prev_food_distance:  # Not moving away from food
                        enhanced_reward += self.config.tactical_patience_bonus * 0.02
                
                self._prev_food_distance = food_distance
        
        return enhanced_reward
    
    def update_performance_metrics(self, final_score: int):
        """Update performance tracking metrics"""
        self.recent_scores.append(final_score)
        self.session_stats["episodes_played"] += 1
        
        # Update score brackets
        if final_score >= 15:
            self.session_stats["scores_15_plus"] += 1
        if final_score >= 20:
            self.session_stats["scores_20_plus"] += 1
        if final_score >= 25:
            self.session_stats["scores_25_plus"] += 1
        if final_score >= 30:
            self.session_stats["scores_30_plus"] += 1
        
        # Update consistency tracking
        if final_score >= self.config.performance_consistency_threshold:
            self.consistency_streak += 1
        else:
            self.consistency_streak = 0
        
        # Update best score
        if final_score > self.best_intermediate_score:
            self.best_intermediate_score = final_score
        
        # Calculate consistency rate
        if len(self.recent_scores) >= 10:
            consistent_scores = sum(1 for score in self.recent_scores if score >= 15)
            self.session_stats["consistency_rate"] = consistent_scores / len(self.recent_scores)
        
        # Calculate average score
        if self.recent_scores:
            self.session_stats["avg_score_last_20"] = np.mean(self.recent_scores)
    
    def get_training_recommendations(self) -> List[str]:
        """Get training recommendations based on current performance"""
        recommendations = []
        
        if self.session_stats["consistency_rate"] < 0.3:
            recommendations.append("Focus on consistent 15+ point performance")
        
        if self.session_stats["scores_20_plus"] == 0 and self.session_stats["episodes_played"] > 10:
            recommendations.append("Work on breaking through the 20-point barrier")
        
        if self.consistency_streak < 3:
            recommendations.append("Improve consistency in intermediate score range")
        
        if self.session_stats["avg_score_last_20"] < 18:
            recommendations.append("Target average score improvement to 18+")
        
        return recommendations
    
    def print_progress_report(self):
        """Print detailed progress report for intermediate training"""
        print(f"\n📊 Intermediate Training Progress Report - Snake {self.snake_id}")
        print(f"{'='*60}")
        print(f"Episodes Played: {self.session_stats['episodes_played']}")
        print(f"Best Intermediate Score: {self.best_intermediate_score}")
        print(f"Current Consistency Streak: {self.consistency_streak}")
        print(f"Average Score (Last 20): {self.session_stats['avg_score_last_20']:.1f}")
        print(f"Consistency Rate (15+): {self.session_stats['consistency_rate']:.1%}")
        print(f"")
        print(f"Score Distribution:")
        print(f"  15+ points: {self.session_stats['scores_15_plus']} episodes")
        print(f"  20+ points: {self.session_stats['scores_20_plus']} episodes")
        print(f"  25+ points: {self.session_stats['scores_25_plus']} episodes")
        print(f"  30+ points: {self.session_stats['scores_30_plus']} episodes")
        print(f"")
        print(f"Skill Milestones:")
        for milestone, achieved in self.skill_milestones.items():
            status = "✅" if achieved else "❌"
            print(f"  {status} {milestone}")
        
        recommendations = self.get_training_recommendations()
        if recommendations:
            print(f"\n🎯 Training Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        print(f"{'='*60}")
    
    def save_intermediate_training_stats(self):
        """Save training statistics"""
        stats_data = {
            "snake_id": self.snake_id,
            "session_stats": self.session_stats,
            "recent_scores": list(self.recent_scores),
            "consistency_streak": self.consistency_streak,
            "best_intermediate_score": self.best_intermediate_score,
            "skill_milestones": self.skill_milestones,
            "config": {
                "replay_score_range": self.config.replay_score_range,
                "performance_consistency_threshold": self.config.performance_consistency_threshold
            }
        }
        
        try:
            with open(self.config.stats_save_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
            print(f"💾 Intermediate training stats saved")
        except Exception as e:
            print(f"❌ Failed to save stats: {e}")

def create_intermediate_training_session(snake_id: str = "A") -> IntermediateScoreTrainer:
    """Create a specialized training session for intermediate score development"""
    config = IntermediateTrainingConfig()
    trainer = IntermediateScoreTrainer(snake_id, config)
    
    print(f"🎯 Created Intermediate Score Trainer for Snake {snake_id}")
    print(f"Target Score Range: {config.replay_score_range[0]}-{config.replay_score_range[1]} points")
    print(f"Loaded {len(trainer.replay_episodes)} replay episodes for learning")
    print(f"Performance Threshold: {config.performance_consistency_threshold}+ points")
    
    return trainer

if __name__ == "__main__":
    # Test the intermediate training system
    print("🧠 Testing Intermediate Score Training System")
    
    # Create trainer
    trainer = create_intermediate_training_session("A")
    
    # Print initial status
    trainer.print_progress_report()
    
    # Simulate some training episodes
    print("\n🎮 Simulating training episodes...")
    
    # Simulate performance in intermediate range
    test_scores = [16, 19, 22, 14, 18, 21, 26, 15, 23, 29, 17, 20, 24, 31, 18]
    
    for i, score in enumerate(test_scores):
        trainer.update_performance_metrics(score)
        
        if i % 5 == 4:  # Print progress every 5 episodes
            print(f"\nAfter {i+1} episodes:")
            trainer.print_progress_report()
    
    # Save final stats
    trainer.save_intermediate_training_stats()
    
    print("✅ Intermediate training system test completed!")