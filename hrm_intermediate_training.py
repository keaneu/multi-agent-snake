"""
HRM Intermediate Training System (15-30 Score Range)
Builds upon trained foundation with enhanced reward structure and hierarchical reasoning
Uses replays folder for 15-30 score range specialized training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ai_training_agent import AITrainingAgent, TrainingConfig
from hrm_enhanced_dqn import HRMEnhancedDQN, HRMReplayInformedTrainer, create_hrm_enhanced_dqn
from hrm_system import GoalType

@dataclass
class HRMIntermediateConfig(TrainingConfig):
    """Enhanced HRM configuration for intermediate score development (15-30 range)"""
    
    # Core HRM Training Parameters
    hidden_size: int = 128  # Match foundation model architecture
    learning_rate: float = 0.0007  # Reduced for fine-tuning on foundation
    gamma: float = 0.97  # Enhanced long-term planning
    
    # Foundation Model Loading
    foundation_model_path: str = "snake_ai_model_A.pth"  # Pre-trained 10-30 foundation
    load_foundation: bool = True
    freeze_foundation_layers: bool = False  # Allow fine-tuning of all layers
    
    # Intermediate-Specific Exploration (reduced from foundation)
    epsilon_start: float = 0.3  # Start lower since we have foundation
    epsilon_end: float = 0.015
    epsilon_decay: float = 0.997
    
    # Enhanced Experience Replay for Intermediate Range
    memory_size: int = 20000
    batch_size: int = 64
    target_update_freq: int = 60
    train_freq: int = 2
    
    # HRM-Specific Parameters for Intermediate Training
    use_hrm: bool = True
    hrm_hierarchical_loss_weight: float = 0.25
    hrm_goal_value_loss_weight: float = 0.2
    hrm_intermediate_focus_weight: float = 0.3  # New: Focus on intermediate patterns
    
    # INTERMEDIATE REWARD STRUCTURE (15-30 range optimization)
    # Base Rewards - Refined for consistent intermediate performance
    food_reward: float = 12.0           # Enhanced food reward for progression
    death_penalty: float = -5.0         # Reduced penalty for calculated risk-taking
    step_penalty: float = -0.0008       # Minimal step penalty for patience
    survival_reward: float = 0.18       # Enhanced survival for longer episodes
    
    # Intermediate Range Strategic Rewards
    intermediate_consistency_reward: float = 2.5    # Reward for 10+ consistency
    breakthrough_momentum_bonus: float = 4.0        # Bonus for 15+ breakthrough
    consolidation_mastery_bonus: float = 3.0        # Bonus for 20+ consolidation
    advanced_tactics_bonus: float = 5.0             # Bonus for 25+ advanced play
    
    # HRM-Enhanced Strategic Bonuses
    hierarchical_decision_bonus: float = 1.0        # Bonus for good HRM decisions
    goal_completion_bonus: float = 2.0              # Bonus for achieving HRM goals
    multi_level_reasoning_bonus: float = 1.5        # Bonus for using multiple reasoning levels
    adaptive_strategy_bonus: float = 1.2            # Bonus for strategy adaptation
    
    # Enhanced Tactical Rewards for Intermediate Play
    efficient_pathfinding_reward: float = 0.8       # Reward for optimal paths
    risk_assessment_bonus: float = 0.6              # Bonus for good risk evaluation
    territorial_awareness_bonus: float = 0.7        # Bonus for space management
    timing_optimization_bonus: float = 0.5          # Bonus for good timing decisions
    
    # Progressive Enhancement (builds on foundation)
    foundation_skill_bonus: float = 1.0             # Bonus for demonstrating foundation skills
    skill_transfer_bonus: float = 0.8               # Bonus for transferring learned patterns
    intermediate_specialization_bonus: float = 1.5  # Bonus for specializing in 15-30 range
    
    # Replay System Configuration for 10-25 Range
    replay_score_range: Tuple[int, int] = (10, 25)
    replay_data_path: str = "replays"
    foundation_replay_weight: float = 0.3  # Weight for foundation replay patterns
    intermediate_replay_weight: float = 0.7  # Weight for intermediate replay patterns
    
    # Model Configuration
    model_save_path: str = "snake_ai_hrm_intermediate_15_30.pth"
    stats_save_path: str = "hrm_intermediate_training_stats.json"

class HRMIntermediateTrainer:
    """
    Specialized HRM trainer for intermediate score development
    Builds upon pre-trained foundation model with enhanced hierarchical reasoning
    """
    
    def __init__(self, snake_id: str, config: Optional[HRMIntermediateConfig] = None):
        self.config = config or HRMIntermediateConfig()
        self.snake_id = snake_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize HRM-Enhanced DQN
        self.model = create_hrm_enhanced_dqn(
            input_size=33,
            hidden_size=self.config.hidden_size,
            output_size=3
        ).to(self.device)
        
        # Load foundation model if specified
        self.foundation_loaded = False
        if self.config.load_foundation and os.path.exists(self.config.foundation_model_path):
            self.load_foundation_model()
        
        # Initialize specialized trainer
        self.trainer = HRMReplayInformedTrainer(self.model, self.config.learning_rate)
        self.trainer.hierarchical_loss_weight = self.config.hrm_hierarchical_loss_weight
        self.trainer.goal_value_loss_weight = self.config.hrm_goal_value_loss_weight
        
        # Intermediate performance tracking
        self.performance_metrics = {
            "episodes_trained": 0,
            "intermediate_scores": deque(maxlen=50),
            "breakthrough_count": 0,  # 20+ scores
            "consolidation_count": 0,  # 25+ scores
            "mastery_count": 0,       # 30+ scores
            "consistency_streak": 0,   # Consecutive 10+ scores
            "best_score": 0,
            "hrm_decision_quality": deque(maxlen=100),
            "goal_completion_rate": deque(maxlen=100)
        }
        
        # Load intermediate replays
        self.intermediate_replays = []
        self.load_intermediate_replays()
        
        # HRM-specific tracking
        self.hrm_session_metrics = {
            "hierarchical_decisions": 0,
            "goal_alignments": 0,
            "strategic_overrides": 0,
            "tactical_executions": 0,
            "meta_level_activations": 0
        }
        
        print(f"🧠🎯 HRM Intermediate Trainer initialized for Snake {snake_id}")
        print(f"   Foundation loaded: {self.foundation_loaded}")
        print(f"   Target range: {self.config.replay_score_range}")
        print(f"   Intermediate replays: {len(self.intermediate_replays)}")
    
    def load_foundation_model(self):
        """Load pre-trained foundation model (10-30 range)"""
        try:
            checkpoint = torch.load(self.config.foundation_model_path, map_location=self.device, weights_only=False)
            
            # Load model state dict with compatibility handling
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                model_state = checkpoint
            
            # Load compatible layers only (handle architecture differences)
            model_dict = self.model.state_dict()
            compatible_state = {}
            
            for key, value in model_state.items():
                if key in model_dict and model_dict[key].shape == value.shape:
                    compatible_state[key] = value
                elif key.startswith('enhanced_dqn.') and key in model_dict:
                    # Handle enhanced_dqn component loading
                    compatible_state[key] = value
            
            self.model.load_state_dict(compatible_state, strict=False)
            self.foundation_loaded = True
            
            # Optionally freeze foundation layers
            if self.config.freeze_foundation_layers:
                for name, param in self.model.named_parameters():
                    if 'enhanced_dqn' in name:
                        param.requires_grad = False
            
            print(f"✅ Foundation model loaded: {len(compatible_state)} layers")
            
        except Exception as e:
            print(f"⚠️ Could not load foundation model: {e}")
            self.foundation_loaded = False
    
    def load_intermediate_replays(self):
        """Load replay episodes specifically in 15-30 range"""
        if not os.path.exists(self.config.replay_data_path):
            print(f"⚠️ Replay path not found: {self.config.replay_data_path}")
            return
        
        target_min, target_max = self.config.replay_score_range
        replay_files = []
        
        for filename in os.listdir(self.config.replay_data_path):
            if filename.endswith('.json') and 'score_' in filename:
                try:
                    score_part = filename.split('score_')[1].split('.')[0]
                    score = int(score_part)
                    
                    if target_min <= score <= target_max:
                        replay_files.append((os.path.join(self.config.replay_data_path, filename), score))
                except (ValueError, IndexError):
                    continue
        
        # Sort by score for balanced sampling
        replay_files.sort(key=lambda x: x[1])
        
        # Load replay data with score distribution balancing
        score_buckets = defaultdict(list)
        for filepath, score in replay_files:
            bucket = (score // 5) * 5  # Group into 5-point buckets
            score_buckets[bucket].append(filepath)
        
        # Load balanced samples from each bucket
        for bucket, files in score_buckets.items():
            sample_size = min(10, len(files))  # Max 10 per bucket
            sampled_files = random.sample(files, sample_size)
            
            for filepath in sampled_files:
                try:
                    with open(filepath, 'r') as f:
                        replay_data = json.load(f)
                        self.intermediate_replays.append(replay_data)
                except Exception as e:
                    print(f"⚠️ Error loading replay {filepath}: {e}")
        
        print(f"📈 Loaded {len(self.intermediate_replays)} intermediate replays")
        
        # Print distribution
        score_dist = defaultdict(int)
        for replay in self.intermediate_replays:
            score = replay.get('final_score', 0)
            bucket = (score // 5) * 5
            score_dist[bucket] += 1
        
        print(f"   Score distribution: {dict(score_dist)}")
    
    def calculate_hrm_intermediate_rewards(self, base_reward: float, event_type: str, 
                                         game_data: Dict[str, Any], hrm_explanation: Dict[str, Any]) -> float:
        """Calculate enhanced rewards using HRM reasoning for intermediate range"""
        enhanced_reward = base_reward
        
        # Get current game state info
        current_score = game_data.get('current_score', 0)
        snake_length = game_data.get('snake_length', 3)
        
        # 1. Intermediate Consistency Reward (10+ range)
        if event_type == "food_eaten" and snake_length >= 4:  # ~10 score
            consistency_bonus = self.config.intermediate_consistency_reward
            
            # Scale based on consistency streak
            if self.performance_metrics["consistency_streak"] >= 3:
                consistency_bonus *= 1.5
            elif self.performance_metrics["consistency_streak"] >= 5:
                consistency_bonus *= 2.0
                
            enhanced_reward += consistency_bonus
        
        # 2. Breakthrough Momentum Bonus (15+ breakthrough)
        if event_type == "food_eaten" and snake_length >= 6:  # ~15 score
            if not hasattr(self, '_reached_15_this_episode'):
                self._reached_15_this_episode = True
                enhanced_reward += self.config.breakthrough_momentum_bonus
                self.performance_metrics["breakthrough_count"] += 1
        
        # 3. Consolidation Mastery Bonus (20+ consolidation)
        if event_type == "food_eaten" and snake_length >= 8:  # ~20 score
            if not hasattr(self, '_reached_20_this_episode'):
                self._reached_20_this_episode = True
                enhanced_reward += self.config.consolidation_mastery_bonus
                self.performance_metrics["consolidation_count"] += 1
        
        # 4. Advanced Tactics Bonus (25+ mastery)
        if event_type == "food_eaten" and snake_length >= 10:  # ~25 score
            if not hasattr(self, '_reached_25_this_episode'):
                self._reached_25_this_episode = True
                enhanced_reward += self.config.advanced_tactics_bonus
                self.performance_metrics["mastery_count"] += 1
        
        # 5. HRM-Enhanced Strategic Bonuses
        if hrm_explanation.get('hierarchical_reasoning', False):
            # Hierarchical Decision Quality Bonus
            decision_type = hrm_explanation.get('decision_type', '')
            if decision_type in ['hrm_aligned', 'hrm_strategic']:
                enhanced_reward += self.config.hierarchical_decision_bonus
                self.hrm_session_metrics["hierarchical_decisions"] += 1
            
            # Goal Completion Tracking
            active_goals = hrm_explanation.get('active_goals', [])
            if active_goals and event_type == "food_eaten":
                # Check if food acquisition aligns with active goals
                if any('food' in goal for goal in active_goals):
                    enhanced_reward += self.config.goal_completion_bonus
                    self.performance_metrics["goal_completion_rate"].append(1.0)
                    self.hrm_session_metrics["goal_alignments"] += 1
                else:
                    self.performance_metrics["goal_completion_rate"].append(0.0)
            
            # Multi-level Reasoning Bonus
            hierarchical_values = hrm_explanation.get('hierarchical_values', {})
            if hierarchical_values:
                meta_val = hierarchical_values.get('meta', 0)
                strategic_val = hierarchical_values.get('strategic', 0)
                tactical_val = hierarchical_values.get('tactical', 0)
                
                # Bonus for using multiple reasoning levels effectively
                if abs(meta_val) > 0.1 and abs(strategic_val) > 0.1:
                    enhanced_reward += self.config.multi_level_reasoning_bonus * 0.1
                    self.hrm_session_metrics["meta_level_activations"] += 1
        
        # 6. Enhanced Tactical Rewards for Intermediate Play
        if event_type == "survival" and snake_length >= 5:
            # Efficient Pathfinding Reward
            if hasattr(self, '_track_movement_efficiency'):
                if self._track_movement_efficiency():
                    enhanced_reward += self.config.efficient_pathfinding_reward * 0.05
            
            # Risk Assessment Bonus
            dangers = game_data.get('danger_count', 0)
            if dangers <= 1 and snake_length >= 8:  # Good risk management in mid-game
                enhanced_reward += self.config.risk_assessment_bonus * 0.1
        
        # 7. Foundation Skill Transfer Bonuses
        if self.foundation_loaded and event_type in ["food_eaten", "survival"]:
            # Foundation Skill Demonstration
            if snake_length >= 4:  # Demonstrating basic foundation skills
                enhanced_reward += self.config.foundation_skill_bonus * 0.05
            
            # Skill Transfer to Intermediate Range
            if 6 <= snake_length <= 10:  # Intermediate range skill application
                enhanced_reward += self.config.skill_transfer_bonus * 0.1
            
            # Intermediate Specialization
            if snake_length >= 8:  # Specializing in intermediate performance
                enhanced_reward += self.config.intermediate_specialization_bonus * 0.08
        
        return enhanced_reward
    
    def train_step_with_hrm_intermediate(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Enhanced training step with HRM intermediate focus"""
        if len(experiences) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(experiences, self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Standard HRM training step
        loss_dict = self.trainer.train_step(states, actions, rewards, next_states, dones, self.config.gamma)
        
        # Add intermediate-specific loss components
        with torch.no_grad():
            current_outputs = self.model.forward(states, use_hrm=False)
            
            # Intermediate range focus loss
            intermediate_mask = self._create_intermediate_mask(batch)
            if intermediate_mask.sum() > 0:
                # Enhanced loss weighting for intermediate score patterns
                intermediate_q_values = current_outputs['q_values'][intermediate_mask]
                intermediate_actions = actions[intermediate_mask]
                intermediate_rewards = rewards[intermediate_mask]
                
                # Calculate intermediate-focused loss
                selected_q_values = intermediate_q_values.gather(1, intermediate_actions.unsqueeze(1))
                intermediate_loss = F.mse_loss(selected_q_values.squeeze(), intermediate_rewards)
                
                # Add to total loss (additional backward pass)
                intermediate_loss_weighted = self.config.hrm_intermediate_focus_weight * intermediate_loss
                
                self.trainer.optimizer.zero_grad()
                intermediate_loss_weighted.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.trainer.optimizer.step()
                
                loss_dict['intermediate_focus_loss'] = intermediate_loss.item()
        
        return loss_dict
    
    def _create_intermediate_mask(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Create mask for experiences in intermediate score range"""
        mask = []
        for exp in batch:
            # Estimate score from snake length or other indicators
            estimated_score = exp.get('estimated_score', 0)
            if estimated_score == 0:
                # Fallback estimation
                state = exp['state']
                snake_length = self._estimate_snake_length_from_state(state)
                estimated_score = max(0, (snake_length - 3) * 1)  # Rough estimate
            
            is_intermediate = 15 <= estimated_score <= 30
            mask.append(is_intermediate)
        
        return torch.BoolTensor(mask).to(self.device)
    
    def _estimate_snake_length_from_state(self, state: np.ndarray) -> int:
        """Estimate snake length from state vector"""
        # Assuming state[2] contains normalized snake length
        if len(state) > 2:
            normalized_length = state[2]
            estimated_length = int(normalized_length * 50)  # Denormalize
            return max(3, estimated_length)
        return 3
    
    def update_performance_metrics(self, final_score: int, episode_data: Dict[str, Any]):
        """Update performance tracking for intermediate training"""
        self.performance_metrics["episodes_trained"] += 1
        self.performance_metrics["intermediate_scores"].append(final_score)
        
        if final_score > self.performance_metrics["best_score"]:
            self.performance_metrics["best_score"] = final_score
        
        # Update consistency streak
        if final_score >= 15:
            self.performance_metrics["consistency_streak"] += 1
        else:
            self.performance_metrics["consistency_streak"] = 0
        
        # Reset episode-specific flags
        if hasattr(self, '_reached_15_this_episode'):
            delattr(self, '_reached_15_this_episode')
        if hasattr(self, '_reached_20_this_episode'):
            delattr(self, '_reached_20_this_episode')
        if hasattr(self, '_reached_25_this_episode'):
            delattr(self, '_reached_25_this_episode')
        
        # Update HRM decision quality
        hrm_decisions = episode_data.get('hrm_decisions', 0)
        total_decisions = episode_data.get('total_decisions', 1)
        decision_quality = hrm_decisions / max(total_decisions, 1)
        self.performance_metrics["hrm_decision_quality"].append(decision_quality)
    
    def get_training_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive training progress report"""
        recent_scores = list(self.performance_metrics["intermediate_scores"])
        
        report = {
            "episodes_trained": self.performance_metrics["episodes_trained"],
            "best_score": self.performance_metrics["best_score"],
            "consistency_streak": self.performance_metrics["consistency_streak"],
            "foundation_loaded": self.foundation_loaded,
            
            # Score Analysis
            "average_score_last_20": np.mean(recent_scores[-20:]) if len(recent_scores) >= 20 else np.mean(recent_scores),
            "score_improvement_trend": self._calculate_score_trend(recent_scores),
            
            # Milestone Achievements
            "breakthrough_count": self.performance_metrics["breakthrough_count"],
            "consolidation_count": self.performance_metrics["consolidation_count"],
            "mastery_count": self.performance_metrics["mastery_count"],
            
            # HRM Performance
            "avg_hrm_decision_quality": np.mean(self.performance_metrics["hrm_decision_quality"]) if self.performance_metrics["hrm_decision_quality"] else 0,
            "avg_goal_completion_rate": np.mean(self.performance_metrics["goal_completion_rate"]) if self.performance_metrics["goal_completion_rate"] else 0,
            "hrm_session_metrics": dict(self.hrm_session_metrics),
            
            # Score Distribution
            "score_distribution": self._analyze_score_distribution(recent_scores)
        }
        
        return report
    
    def _calculate_score_trend(self, scores: List[int]) -> str:
        """Calculate score improvement trend"""
        if len(scores) < 10:
            return "insufficient_data"
        
        recent_10 = scores[-10:]
        previous_10 = scores[-20:-10] if len(scores) >= 20 else scores[:-10]
        
        recent_avg = np.mean(recent_10)
        previous_avg = np.mean(previous_10)
        
        if recent_avg > previous_avg + 2:
            return "improving"
        elif recent_avg < previous_avg - 2:
            return "declining"
        else:
            return "stable"
    
    def _analyze_score_distribution(self, scores: List[int]) -> Dict[str, int]:
        """Analyze score distribution across ranges"""
        distribution = {
            "10-14": 0, "15-19": 0, "20-24": 0, 
            "25-29": 0, "30-34": 0, "35+": 0
        }
        
        for score in scores:
            if score < 15:
                distribution["10-14"] += 1
            elif score < 20:
                distribution["15-19"] += 1
            elif score < 25:
                distribution["20-24"] += 1
            elif score < 30:
                distribution["25-29"] += 1
            elif score < 35:
                distribution["30-34"] += 1
            else:
                distribution["35+"] += 1
        
        return distribution
    
    def save_intermediate_model(self):
        """Save intermediate training model and metrics"""
        try:
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'trainer_state_dict': self.trainer.optimizer.state_dict(),
                'config': self.config.__dict__,
                'performance_metrics': {
                    key: list(value) if isinstance(value, deque) else value
                    for key, value in self.performance_metrics.items()
                },
                'hrm_session_metrics': self.hrm_session_metrics,
                'foundation_loaded': self.foundation_loaded,
                'training_timestamp': time.time()
            }
            
            torch.save(save_data, self.config.model_save_path)
            print(f"💾 HRM Intermediate model saved: {self.config.model_save_path}")
            
            # Save detailed stats
            with open(self.config.stats_save_path, 'w') as f:
                json.dump(self.get_training_progress_report(), f, indent=2)
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        report = self.get_training_progress_report()
        
        print(f"\n🧠🎯 HRM Intermediate Training Summary - Snake {self.snake_id}")
        print(f"{'='*70}")
        print(f"Episodes Trained: {report['episodes_trained']}")
        print(f"Foundation Model: {'✅ Loaded' if report['foundation_loaded'] else '❌ Not loaded'}")
        print(f"Best Score: {report['best_score']}")
        print(f"Consistency Streak (15+): {report['consistency_streak']}")
        print(f"Average Score (Last 20): {report['average_score_last_20']:.1f}")
        print(f"Score Trend: {report['score_improvement_trend']}")
        print(f"")
        print(f"🎯 Milestone Achievements:")
        print(f"  Breakthrough (20+): {report['breakthrough_count']} times")
        print(f"  Consolidation (25+): {report['consolidation_count']} times") 
        print(f"  Mastery (30+): {report['mastery_count']} times")
        print(f"")
        print(f"🧠 HRM Performance:")
        print(f"  Decision Quality: {report['avg_hrm_decision_quality']:.2%}")
        print(f"  Goal Completion Rate: {report['avg_goal_completion_rate']:.2%}")
        print(f"  Hierarchical Decisions: {report['hrm_session_metrics']['hierarchical_decisions']}")
        print(f"  Goal Alignments: {report['hrm_session_metrics']['goal_alignments']}")
        print(f"  Meta-Level Activations: {report['hrm_session_metrics']['meta_level_activations']}")
        print(f"")
        print(f"📊 Score Distribution:")
        for range_name, count in report['score_distribution'].items():
            print(f"  {range_name}: {count} episodes")
        print(f"{'='*70}")

def create_hrm_intermediate_trainer(snake_id: str = "A", foundation_model_path: str = "snake_ai_model_A.pth") -> HRMIntermediateTrainer:
    """Create HRM intermediate trainer with foundation model"""
    config = HRMIntermediateConfig()
    config.foundation_model_path = foundation_model_path
    
    trainer = HRMIntermediateTrainer(snake_id, config)
    
    print(f"🎯🧠 Created HRM Intermediate Trainer for Snake {snake_id}")
    print(f"Foundation: {foundation_model_path}")
    print(f"Target Range: {config.replay_score_range}")
    
    return trainer

if __name__ == "__main__":
    # Test the HRM intermediate training system
    print("🧠🎯 Testing HRM Intermediate Training System")
    
    # Create trainer
    trainer = create_hrm_intermediate_trainer("A")
    
    # Print initial status
    trainer.print_training_summary()
    
    # Simulate some training episodes
    print("\n🎮 Simulating intermediate training episodes...")
    
    # Simulate intermediate-focused performance
    test_scores = [18, 22, 16, 26, 19, 24, 31, 17, 28, 20, 25, 29, 15, 33, 21]
    
    for i, score in enumerate(test_scores):
        episode_data = {
            'hrm_decisions': random.randint(15, 25),
            'total_decisions': random.randint(20, 30)
        }
        
        trainer.update_performance_metrics(score, episode_data)
        
        if i % 5 == 4:  # Print progress every 5 episodes
            print(f"\nAfter {i+1} episodes:")
            trainer.print_training_summary()
    
    # Save final model and stats
    trainer.save_intermediate_model()
    
    print("✅ HRM Intermediate training system test completed!")