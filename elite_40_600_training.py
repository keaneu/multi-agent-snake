"""
Elite 40-600 Score Training - Focused on ultra high-performance replay patterns
Uses same reward structure but trains exclusively on 40-600 point replays
"""

import time
import signal
import sys
import os
import pickle
from collections import deque
from multi_agent_framework import AgentRegistry
from game_engine_agent import GameEngineAgent, GameConfig
from snake_logic_agent import SnakeLogicAgent
from visualization_agent import VisualizationAgent, RenderSettings
from environment_agent import EnvironmentAgent, EnvironmentConfig
from ai_training_agent import AITrainingAgent, TrainingConfig
from replay_agent import ReplayAgent

class Elite40600TrainingSystem:
    """Elite training system focused on 40-600 score replays"""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.agents = {}
        self.running = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Elite 40-600 training shutdown signal received...")
        self.shutdown()
        sys.exit(0)
    
    def create_agents(self):
        """Create agents optimized for elite 40-600 training"""
        print("🏆 Creating Elite 40-600 Training Agents...")
        
        # 1. Game Engine Agent - same configuration
        game_config = GameConfig(
            grid_width=20,
            grid_height=20,
            target_fps=8,  # Slower for better observation
            max_food_count=3,
            points_per_food=10,
            points_per_second=1,
            collision_penalty=-50,
            wrap_walls=False
        )
        self.agents['game_engine'] = GameEngineAgent(game_config)
        
        # 2. Snake Logic Agents
        self.agents['snake_a'] = SnakeLogicAgent("A")
        self.agents['snake_b'] = SnakeLogicAgent("B")
        
        # 3. Environment Agent - same configuration
        env_config = EnvironmentConfig(
            grid_width=20,
            grid_height=20,
            max_food_count=3,
            food_spawn_rate=1.5,
            obstacle_count=4,
            dynamic_obstacles=True,
            food_clustering=False,
            wrap_walls=False
        )
        self.agents['environment'] = EnvironmentAgent(env_config)
        
        # 4. Console Visualization (lightweight)
        from test_multi_agent_system import ConsoleVisualizationAgent
        self.agents['visualization'] = ConsoleVisualizationAgent()
        
        # 5. AI Training Agents - SAME reward structure as original
        ai_config_a = TrainingConfig(
            # Neural network parameters - same
            hidden_size=256,
            learning_rate=0.005,
            gamma=0.95,
            
            # Training parameters - Phase 44 Legendary Survival-First Enhancement (SAME)
            epsilon_start=0.1,
            epsilon_end=0.05,
            epsilon_decay=0.9999,
            
            # Experience replay - Enhanced Elite Memory (30K Buffer)
            memory_size=30000,
            batch_size=128,
            target_update_freq=100,
            train_freq=1,
            save_freq=50,
            
            # Dense rewards - EXACT SAME structure
            use_dense_rewards=True,
            food_reward=5000000000.0,
            death_penalty=0.0,
            step_penalty=0.0,
            survival_reward=50000000.0,
            
            # Phase 44: Legendary Ascension Rewards - SAME
            food_proximity_reward=3000000000.0,
            wall_avoidance_reward=0.0,
            efficient_movement_reward=6000000000.0,
            exploration_reward=1500000000.0,
            length_bonus_per_segment=3000000000.0,
            length_progression_reward=10000000000.0,
            safe_exploration_reward=2500000000.0,
            strategic_positioning_reward=6000000000.0,
            
            # Phase 44: Legendary Ascension Milestones - SAME
            legendary_60_bonus=100000000000000.0,
            ascension_100_bonus=500000000000000.0,
            mythical_200_bonus=2500000000000000.0,
            godmode_400_bonus=10000000000000000.0,
            ultra_sequence_bonus=40.0,
            mastery_growth_bonus=50.0,
            ultra_patience_bonus=2.0,
            
            # All other bonuses - SAME
            territorial_supremacy_bonus=3.2,
            multi_food_legend_bonus=4.5,
            calculated_aggression_bonus=2.8,
            championship_endurance_bonus=2.4,
            territory_control_bonus=0.8,
            competitive_advantage_bonus=1.5,
            efficiency_mastery_bonus=1.2,
            progressive_food_bonus=0.7,
            tactical_movement_bonus=0.4,
            skill_development_bonus=0.3,
            efficiency_improvement_bonus=0.5,
            patience_bonus=0.1,
            
            # Model persistence
            model_save_path="snake_ai_elite_40_600_A.pth",
            stats_save_path="elite_40_600_training_stats.json",
            memory_save_path="elite_40_600_memory_A.pkl",
            
            # HRM parameters - SAME
            use_hrm=True,
            hrm_hierarchical_loss_weight=0.3,
            hrm_goal_value_loss_weight=0.2,
            
            # Enhanced Exploration Parameters - SAME
            enhanced_exploration=True,
            epsilon_boost=0.0,
            adaptive_epsilon=True,
            plateau_threshold=10,
            exploration_schedule="adaptive_breakthrough"
        )
        
        ai_config_b = TrainingConfig(
            # Same configuration as A but different model path
            hidden_size=256,
            learning_rate=0.005,
            gamma=0.95,
            epsilon_start=0.1,
            epsilon_end=0.05,
            epsilon_decay=0.9999,
            memory_size=30000,
            batch_size=128,
            target_update_freq=100,
            train_freq=1,
            save_freq=50,
            use_dense_rewards=True,
            food_reward=5000000000.0,
            death_penalty=0.0,
            step_penalty=0.0,
            survival_reward=50000000.0,
            food_proximity_reward=3000000000.0,
            wall_avoidance_reward=0.0,
            efficient_movement_reward=6000000000.0,
            exploration_reward=1500000000.0,
            length_bonus_per_segment=3000000000.0,
            length_progression_reward=10000000000.0,
            safe_exploration_reward=2500000000.0,
            strategic_positioning_reward=6000000000.0,
            legendary_60_bonus=100000000000000.0,
            ascension_100_bonus=500000000000000.0,
            mythical_200_bonus=2500000000000000.0,
            godmode_400_bonus=10000000000000000.0,
            ultra_sequence_bonus=40.0,
            mastery_growth_bonus=50.0,
            ultra_patience_bonus=2.0,
            territorial_supremacy_bonus=3.2,
            multi_food_legend_bonus=4.5,
            calculated_aggression_bonus=2.8,
            championship_endurance_bonus=2.4,
            territory_control_bonus=0.8,
            competitive_advantage_bonus=1.5,
            efficiency_mastery_bonus=1.2,
            progressive_food_bonus=0.7,
            tactical_movement_bonus=0.4,
            skill_development_bonus=0.3,
            efficiency_improvement_bonus=0.5,
            patience_bonus=0.1,
            model_save_path="snake_ai_elite_40_600_B.pth",
            stats_save_path="elite_40_600_training_stats_b.json",
            memory_save_path="elite_40_600_memory_B.pkl",
            use_hrm=True,
            hrm_hierarchical_loss_weight=0.3,
            hrm_goal_value_loss_weight=0.2,
            enhanced_exploration=True,
            epsilon_boost=0.0,
            adaptive_epsilon=True,
            plateau_threshold=10,
            exploration_schedule="adaptive_breakthrough"
        )
        
        self.agents['ai_a'] = AITrainingAgent("A", ai_config_a)
        self.agents['ai_b'] = AITrainingAgent("B", ai_config_b)
        
        # Load existing memory buffers if available
        self.load_memory_buffers()
        
        # 6. Replay Agent - ELITE FOCUSED (40-600 score range)
        self.agents['replay'] = ReplayAgent(high_score_threshold=40, max_score_threshold=600)

        print(f"✅ Created {len(self.agents)} elite training agents")
        print("🎯 Focus: 40-600 score range with SAME reward structure")
    
    def register_agents(self):
        """Register all agents"""
        print("📝 Registering elite training agents...")
        
        for agent_name, agent in self.agents.items():
            self.registry.register_agent(agent)
            print(f"  ✓ {agent_name}: {agent.agent_id}")
        
        print("✅ All agents registered")
    
    def start_system(self):
        """Start the elite training system"""
        print("\n🚀 Starting Elite 40-600 Training System...")
        
        # Start all agents
        self.registry.start_all_agents()
        self.running = True
        
        # Wait for initialization
        time.sleep(2)
        
        print("✅ Elite training system started!")
        print("\n" + "="*60)
        print("🏆 ELITE 40-600 SCORE TRAINING ACTIVE")
        print("="*60)
        print("Target: Learn from 40-600 point replays")
        print("Reward Structure: Same as Phase 44 Legendary")
        print("Duration: 1650 seconds")
        print("="*60)
    
    def run_training(self, duration: int = 1650):
        """Run training for specified duration"""
        print(f"\n🎯 Elite training for {duration} seconds...")
        
        start_time = time.time()
        last_report = start_time
        report_interval = 30  # Report every 30 seconds
        
        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = duration - elapsed
                
                # Report progress periodically
                if current_time - last_report >= report_interval:
                    print(f"\n⏱️  Elite Training Progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
                    self.print_elite_stats()
                    last_report = current_time
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n⏹️  Elite training interrupted")
    
    def print_elite_stats(self):
        """Print elite training statistics"""
        print("📊 ELITE TRAINING STATUS:")
        
        # AI Training Stats
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                state = ai_agent.get_public_state()
                print(f"🐍 Elite Snake {snake_id}:")
                print(f"   Episode: {state.get('episode', 0)}")
                print(f"   Epsilon: {state.get('epsilon', 0):.3f}")
                print(f"   Avg Reward: {state.get('avg_reward', 0):.1f}")
                print(f"   Network: {state.get('network_type', 'Unknown')}")
    
    def shutdown(self):
        """Gracefully shutdown"""
        if not self.running:
            return
            
        print("\n🔄 Shutting down elite training...")
        self.running = False
        
        # Stop all agents
        self.registry.stop_all_agents()
        
        # Save models and memory buffers
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                try:
                    ai_agent.save_model()
                    print(f"💾 Saved elite model for snake {snake_id}")
                    self.save_memory_buffer(ai_agent, snake_id)
                except Exception as e:
                    print(f"⚠️  Could not save elite model/memory: {e}")
        
        print("✅ Elite training shutdown complete")

    def save_memory_buffer(self, ai_agent, snake_id):
        """Save the experience replay buffer to preserve across restarts"""
        try:
            if hasattr(ai_agent, 'memory') and hasattr(ai_agent.config, 'memory_save_path'):
                memory_path = ai_agent.config.memory_save_path
                memory_data = {
                    'memory': list(ai_agent.memory),  # Convert deque to list
                    'episode_count': ai_agent.episode_count,
                    'training_step': ai_agent.training_step,
                    'epsilon': ai_agent.epsilon
                }
                
                with open(memory_path, 'wb') as f:
                    pickle.dump(memory_data, f)
                print(f"💾 Saved memory buffer for snake {snake_id}: {len(ai_agent.memory)} experiences")
        except Exception as e:
            print(f"⚠️ Could not save memory buffer for snake {snake_id}: {e}")
    
    def load_memory_buffers(self):
        """Load existing memory buffers to continue training"""
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent and hasattr(ai_agent.config, 'memory_save_path'):
                try:
                    memory_path = ai_agent.config.memory_save_path
                    if os.path.exists(memory_path):
                        with open(memory_path, 'rb') as f:
                            memory_data = pickle.load(f)
                        
                        # Restore memory buffer
                        ai_agent.memory = deque(memory_data['memory'], maxlen=ai_agent.config.memory_size)
                        ai_agent.episode_count = memory_data.get('episode_count', 0)
                        ai_agent.training_step = memory_data.get('training_step', 0)
                        ai_agent.epsilon = memory_data.get('epsilon', ai_agent.config.epsilon_start)
                        
                        print(f"🔄 Restored memory buffer for snake {snake_id}: {len(ai_agent.memory)} experiences")
                        print(f"   Episode: {ai_agent.episode_count}, Training Steps: {ai_agent.training_step}, Epsilon: {ai_agent.epsilon:.4f}")
                    else:
                        print(f"📝 No existing memory buffer found for snake {snake_id} - starting fresh")
                except Exception as e:
                    print(f"⚠️ Could not load memory buffer for snake {snake_id}: {e}")
                    print(f"📝 Starting with fresh memory buffer")

def main():
    """Main elite training function"""
    print("🏆 Elite 40-600 Score Training System")
    print("=" * 40)
    
    # Create elite training system
    elite_system = Elite40600TrainingSystem()
    
    try:
        # Setup
        elite_system.create_agents()
        elite_system.register_agents()
        
        # Start system
        elite_system.start_system()
        
        # Run for 1650 seconds
        elite_system.run_training(duration=1650)
        
    except Exception as e:
        print(f"❌ Elite training error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Shutdown
        elite_system.shutdown()
        print("\n🎯 Elite 40-600 training session completed!")

if __name__ == "__main__":
    main()