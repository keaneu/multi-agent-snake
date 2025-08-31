"""
HRM Intermediate Training Runner (15-30 Score Range)
Main training script that uses the HRM system with foundation model and 15-30 score replays
"""

import numpy as np
import torch
import time
import json
import os
import sys
import random
from collections import deque
from typing import Dict, Any, List, Optional

from hrm_intermediate_training import HRMIntermediateTrainer, HRMIntermediateConfig, create_hrm_intermediate_trainer
from ai_training_agent import AITrainingAgent, Action
from multi_agent_framework import AgentRegistry, MessageType, Message
from game_engine_agent import GameEngineAgent
from environment_agent import EnvironmentAgent
from snake_logic_agent import SnakeLogicAgent

class HRMIntermediateTrainingSession:
    """
    Complete training session for HRM-based intermediate score development
    Integrates with the multi-agent framework for realistic training environment
    """
    
    def __init__(self, config: Optional[HRMIntermediateConfig] = None):
        self.config = config or HRMIntermediateConfig()
        
        # Training components
        self.trainer = None
        self.registry = AgentRegistry()
        self.training_active = False
        self.current_episode = 0
        self.total_episodes = 0
        
        # Session tracking
        self.session_stats = {
            "start_time": time.time(),
            "episodes_completed": 0,
            "total_training_time": 0,
            "best_session_score": 0,
            "average_score": 0,
            "consistency_improvements": 0,
            "hrm_activation_rate": 0.0
        }
        
        # Episode data collection
        self.episode_data = []
        self.recent_scores = deque(maxlen=100)
        
        print(f"🧠🎯 HRM Intermediate Training Session initialized")
        print(f"Target range: {self.config.replay_score_range}")
        print(f"Foundation model: {self.config.foundation_model_path}")
    
    def setup_training_environment(self):
        """Setup the multi-agent training environment"""
        print("🏗️ Setting up HRM training environment...")
        
        # Create core agents
        game_engine = GameEngineAgent()
        environment = EnvironmentAgent()
        snake_logic_a = SnakeLogicAgent("A")
        snake_logic_b = SnakeLogicAgent("B")  # Optional opponent
        
        # Register agents
        self.registry.register_agent(game_engine)
        self.registry.register_agent(environment)
        self.registry.register_agent(snake_logic_a)
        self.registry.register_agent(snake_logic_b)
        
        # Create HRM trainer (replaces standard AI agent)
        self.trainer = create_hrm_intermediate_trainer("A", self.config.foundation_model_path)
        
        # Create enhanced AI agent that uses the HRM trainer
        self.ai_agent = EnhancedAITrainingAgent("A", self.trainer, self.config)
        self.registry.register_agent(self.ai_agent)
        
        print("✅ Training environment setup complete")
    
    def run_training_session(self, num_episodes: int = 100):
        """Run complete HRM intermediate training session"""
        print(f"🎯 Starting HRM Intermediate Training Session")
        print(f"Episodes: {num_episodes}")
        print(f"Target Score Range: {self.config.replay_score_range[0]}-{self.config.replay_score_range[1]}")
        
        self.total_episodes = num_episodes
        self.training_active = True
        
        try:
            # Start all agents
            self.registry.start_all_agents()
            time.sleep(2)  # Allow agents to initialize
            
            # Training loop
            for episode in range(num_episodes):
                self.current_episode = episode + 1
                
                print(f"\n🎮 Episode {self.current_episode}/{num_episodes}")
                
                # Run single episode
                episode_result = self.run_single_episode()
                
                # Process episode results
                self.process_episode_result(episode_result)
                
                # Print progress
                if self.current_episode % 10 == 0:
                    self.print_training_progress()
                
                # Save progress
                if self.current_episode % 25 == 0:
                    self.save_training_checkpoint()
                
                # Brief pause between episodes
                time.sleep(0.5)
            
            # Final training summary
            self.print_final_training_summary()
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.training_active = False
            self.registry.stop_all_agents()
            
            # Save final model
            if self.trainer:
                self.trainer.save_intermediate_model()
    
    def run_single_episode(self) -> Dict[str, Any]:
        """Run a single training episode"""
        episode_start_time = time.time()
        
        # Reset game
        self.ai_agent.send_message(MessageType.RESET_GAME, "game_engine", {
            "config": {
                "grid_width": 20,
                "grid_height": 20,
                "max_foods": 3,
                "episode_timeout": 300  # 5 minutes max
            }
        })
        
        # Wait for episode completion
        episode_data = self.wait_for_episode_completion()
        
        episode_duration = time.time() - episode_start_time
        episode_data['duration'] = episode_duration
        
        return episode_data
    
    def wait_for_episode_completion(self) -> Dict[str, Any]:
        """Wait for episode to complete and collect data"""
        episode_data = {
            'final_score': 0,
            'episode_length': 0,
            'foods_eaten': 0,
            'hrm_decisions': 0,
            'total_decisions': 0,
            'deaths': 0,
            'completed': False
        }
        
        max_wait_time = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check if episode is complete
            if hasattr(self.ai_agent, 'stats'):
                stats = self.ai_agent.stats
                
                # Episode is complete if game is not active
                if not getattr(self.ai_agent, 'game_active', True):
                    episode_data.update({
                        'final_score': getattr(stats, 'foods_eaten', 0),
                        'episode_length': getattr(stats, 'episode_length', 0),
                        'foods_eaten': getattr(stats, 'foods_eaten', 0),
                        'deaths': getattr(stats, 'deaths', 0),
                        'completed': True
                    })
                    
                    # Get HRM-specific data
                    if hasattr(self.ai_agent, 'hrm_session_data'):
                        hrm_data = self.ai_agent.hrm_session_data
                        episode_data.update({
                            'hrm_decisions': hrm_data.get('hrm_decisions', 0),
                            'total_decisions': hrm_data.get('total_decisions', 0)
                        })
                    
                    break
            
            time.sleep(0.1)  # Check every 100ms
        
        return episode_data
    
    def process_episode_result(self, episode_data: Dict[str, Any]):
        """Process results from completed episode"""
        final_score = episode_data.get('final_score', 0)
        
        # Update session stats
        self.session_stats["episodes_completed"] += 1
        self.recent_scores.append(final_score)
        
        if final_score > self.session_stats["best_session_score"]:
            self.session_stats["best_session_score"] = final_score
        
        if self.recent_scores:
            self.session_stats["average_score"] = np.mean(self.recent_scores)
        
        # Calculate HRM activation rate
        hrm_decisions = episode_data.get('hrm_decisions', 0)
        total_decisions = episode_data.get('total_decisions', 1)
        self.session_stats["hrm_activation_rate"] = hrm_decisions / max(total_decisions, 1)
        
        # Store episode data
        self.episode_data.append(episode_data)
        
        # Update trainer metrics
        if self.trainer:
            self.trainer.update_performance_metrics(final_score, episode_data)
        
        # Check for consistency improvements
        if len(self.recent_scores) >= 10:
            recent_10 = list(self.recent_scores)[-10:]
            consistent_scores = sum(1 for score in recent_10 if score >= 15)
            if consistent_scores >= 7:  # 70% consistency
                self.session_stats["consistency_improvements"] += 1
    
    def print_training_progress(self):
        """Print current training progress"""
        elapsed_time = time.time() - self.session_stats["start_time"]
        
        print(f"\n📊 Training Progress - Episode {self.current_episode}/{self.total_episodes}")
        print(f"{'─'*50}")
        print(f"Elapsed Time: {elapsed_time/60:.1f} minutes")
        print(f"Best Score: {self.session_stats['best_session_score']}")
        print(f"Average Score (Recent): {self.session_stats['average_score']:.1f}")
        print(f"HRM Activation Rate: {self.session_stats['hrm_activation_rate']:.1%}")
        print(f"Consistency Improvements: {self.session_stats['consistency_improvements']}")
        
        # Score distribution for recent episodes
        if len(self.recent_scores) >= 20:
            recent_20 = list(self.recent_scores)[-20:]
            ranges = {
                "10-14": sum(1 for s in recent_20 if 10 <= s <= 14),
                "15-19": sum(1 for s in recent_20 if 15 <= s <= 19),
                "20-24": sum(1 for s in recent_20 if 20 <= s <= 24),
                "25-29": sum(1 for s in recent_20 if 25 <= s <= 29),
                "30+": sum(1 for s in recent_20 if s >= 30)
            }
            print(f"Score Distribution (Last 20): {ranges}")
        
        # Trainer-specific progress
        if self.trainer:
            trainer_report = self.trainer.get_training_progress_report()
            print(f"Trainer Consistency Streak: {trainer_report['consistency_streak']}")
            print(f"Breakthrough Count: {trainer_report['breakthrough_count']}")
            print(f"HRM Decision Quality: {trainer_report['avg_hrm_decision_quality']:.2%}")
    
    def save_training_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_data = {
            'current_episode': self.current_episode,
            'session_stats': self.session_stats,
            'recent_scores': list(self.recent_scores),
            'episode_data': self.episode_data[-50:],  # Last 50 episodes
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        checkpoint_path = f"hrm_training_checkpoint_ep{self.current_episode}.json"
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"💾 Training checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
        
        # Save trainer model
        if self.trainer:
            self.trainer.save_intermediate_model()
    
    def print_final_training_summary(self):
        """Print comprehensive final training summary"""
        total_time = time.time() - self.session_stats["start_time"]
        
        print(f"\n🏆 HRM Intermediate Training Session Complete!")
        print(f"{'='*70}")
        print(f"Total Episodes: {self.session_stats['episodes_completed']}")
        print(f"Total Training Time: {total_time/60:.1f} minutes")
        print(f"Best Session Score: {self.session_stats['best_session_score']}")
        print(f"Final Average Score: {self.session_stats['average_score']:.1f}")
        print(f"Final HRM Activation Rate: {self.session_stats['hrm_activation_rate']:.1%}")
        print(f"Consistency Improvements: {self.session_stats['consistency_improvements']}")
        
        # Detailed analysis
        if self.episode_data:
            scores = [ep['final_score'] for ep in self.episode_data]
            print(f"\n📈 Performance Analysis:")
            print(f"Score Range: {min(scores)} - {max(scores)}")
            print(f"Median Score: {np.median(scores):.1f}")
            print(f"Standard Deviation: {np.std(scores):.1f}")
            
            # Target range analysis
            target_min, target_max = self.config.replay_score_range
            in_target_range = sum(1 for s in scores if target_min <= s <= target_max)
            print(f"Scores in Target Range ({target_min}-{target_max}): {in_target_range}/{len(scores)} ({in_target_range/len(scores):.1%})")
        
        # Trainer summary
        if self.trainer:
            print(f"\n🧠 HRM Trainer Summary:")
            self.trainer.print_training_summary()
        
        print(f"{'='*70}")

class EnhancedAITrainingAgent(AITrainingAgent):
    """Enhanced AI agent that uses HRM intermediate trainer"""
    
    def __init__(self, snake_id: str, hrm_trainer: HRMIntermediateTrainer, config: HRMIntermediateConfig):
        # Initialize with HRM config
        super().__init__(snake_id, config)
        
        # Replace standard components with HRM trainer
        self.hrm_trainer = hrm_trainer
        self.q_network = hrm_trainer.model
        self.trainer = hrm_trainer.trainer
        
        # HRM session tracking
        self.hrm_session_data = {
            'hrm_decisions': 0,
            'total_decisions': 0,
            'episode_start_time': time.time()
        }
    
    def choose_action(self, state: np.ndarray):
        """Enhanced action selection using HRM trainer"""
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
            self.hrm_session_data['total_decisions'] += 1
            
            # Use HRM enhanced action selection
            if hasattr(self.q_network, 'get_action_with_hrm_explanation'):
                state_tensor = torch.FloatTensor(state).to(self.device)
                action_idx, explanation = self.q_network.get_action_with_hrm_explanation(state_tensor, epsilon=0.0)
                
                # Track HRM usage
                if explanation.get('hierarchical_reasoning', False):
                    self.hrm_session_data['hrm_decisions'] += 1
                
                return Action(action_idx)
            else:
                # Fallback to standard action selection
                q_values = self.get_q_values(state)
                return Action(np.argmax(q_values))
    
    def calculate_reward(self, event_type: str, data: Dict[str, Any]) -> float:
        """Enhanced reward calculation using HRM trainer"""
        # Get base reward
        base_reward = super().calculate_reward(event_type, data)
        
        # Get HRM explanation for enhanced rewards
        if hasattr(self, 'last_hrm_explanation'):
            hrm_explanation = self.last_hrm_explanation
        else:
            hrm_explanation = {}
        
        # Use HRM trainer's enhanced reward calculation
        game_data = {
            'current_score': len(self.snake_segments) - 3 if self.snake_segments else 0,
            'snake_length': len(self.snake_segments) if self.snake_segments else 3,
            'danger_count': sum(self.get_danger_detection(self.snake_segments[0])) if self.snake_segments else 0
        }
        
        enhanced_reward = self.hrm_trainer.calculate_hrm_intermediate_rewards(
            base_reward, event_type, game_data, hrm_explanation
        )
        
        return enhanced_reward

def main():
    """Main function to run HRM intermediate training"""
    print("🧠🎯 HRM Intermediate Training System")
    print("=====================================")
    
    # Configuration
    config = HRMIntermediateConfig()
    
    # Check for foundation model
    if not os.path.exists(config.foundation_model_path):
        print(f"⚠️ Foundation model not found: {config.foundation_model_path}")
        print("Please ensure you have a trained foundation model before running intermediate training.")
        return
    
    # Create training session
    session = HRMIntermediateTrainingSession(config)
    
    # Setup environment
    session.setup_training_environment()
    
    # Run training
    try:
        num_episodes = int(input("Enter number of training episodes (default 100): ") or "100")
        session.run_training_session(num_episodes)
    except ValueError:
        print("Invalid input, using default 100 episodes")
        session.run_training_session(100)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("🏁 HRM Intermediate Training Complete!")

if __name__ == "__main__":
    main()