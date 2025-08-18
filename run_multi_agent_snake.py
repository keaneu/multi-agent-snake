"""
Multi-Agent Snake Game - Complete System Integration
Runs all 6 agents together: Game Engine, Snake Logic (A&B), Visualization, Environment, AI Training (A&B)
"""

import time
import signal
import sys
from multi_agent_framework import AgentRegistry
from game_engine_agent import GameEngineAgent, GameConfig
from snake_logic_agent import SnakeLogicAgent
from visualization_agent import VisualizationAgent, RenderSettings
from environment_agent import EnvironmentAgent, EnvironmentConfig
from ai_training_agent import AITrainingAgent, TrainingConfig
from replay_agent import ReplayAgent

class MultiAgentSnakeGame:
    """Complete multi-agent Snake game system"""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.agents = {}
        self.running = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Shutdown signal received...")
        self.shutdown()
        sys.exit(0)
    
    def create_agents(self):
        """Create and configure all agents"""
        print("🔧 Creating agents...")
        
        # 1. Game Engine Agent
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
        
        # 2. Snake Logic Agents (A and B)
        self.agents['snake_a'] = SnakeLogicAgent("A")
        self.agents['snake_b'] = SnakeLogicAgent("B")
        
        # 3. Environment Agent
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
        
        # 4. Visualization Agent (with fallback for macOS)
        viz_config = RenderSettings(
            window_width=800,
            window_height=700,
            cell_size=25,
            show_trail=True,
            show_grid=True,
            show_effects=True
        )
        try:
            self.agents['visualization'] = VisualizationAgent(viz_config)
        except Exception as e:
            print(f"⚠️  Visualization failed, using console mode: {e}")
            # Use console visualization instead
            from test_multi_agent_system import ConsoleVisualizationAgent
            self.agents['visualization'] = ConsoleVisualizationAgent()
        
        # 5. AI Training Agents (A and B)
        ai_config_a = TrainingConfig(
            hidden_size=128,
            learning_rate=0.001,
            epsilon_start=0.9,
            epsilon_decay=0.999,
            memory_size=5000,
            batch_size=32,
            model_save_path="snake_ai_model_A.pth"
        )
        
        ai_config_b = TrainingConfig(
            hidden_size=128,
            learning_rate=0.001,
            epsilon_start=0.9,
            epsilon_decay=0.999,
            memory_size=5000,
            batch_size=32,
            model_save_path="snake_ai_model_B.pth"
        )
        
        self.agents['ai_a'] = AITrainingAgent("A", ai_config_a)
        self.agents['ai_b'] = AITrainingAgent("B", ai_config_b)
        
        # 6. Replay Agent
        self.agents['replay'] = ReplayAgent(high_score_threshold=10)

        print(f"✅ Created {len(self.agents)} agents")
    
    def register_agents(self):
        """Register all agents with the message bus"""
        print("📝 Registering agents...")
        
        for agent_name, agent in self.agents.items():
            self.registry.register_agent(agent)
            print(f"  ✓ {agent_name}: {agent.agent_id}")
        
        print("✅ All agents registered")
    
    def start_system(self):
        """Start the complete multi-agent system"""
        print("\n🚀 Starting multi-agent Snake game system...")
        
        # Start all agents
        self.registry.start_all_agents()
        self.running = True
        
        # Wait a moment for initialization
        time.sleep(2)
        
        print("✅ System started successfully!")
        print("\n" + "="*60)
        print("🎮 MULTI-AGENT SNAKE GAME RUNNING")
        print("="*60)
        print("Controls:")
        print("  G - Toggle grid display")
        print("  T - Toggle snake trails")
        print("  E - Toggle effects")
        print("  R - Reset game")
        print("  Ctrl+C - Shutdown system")
        print("="*60)
    
    def monitor_system(self, duration: int = 60):
        """Monitor the system and report statistics"""
        print(f"\n📊 Monitoring system for {duration} seconds...")
        
        start_time = time.time()
        last_report = start_time
        report_interval = 10  # Report every 10 seconds
        
        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()
                
                # Report statistics periodically
                if current_time - last_report >= report_interval:
                    self.print_system_stats()
                    last_report = current_time
                
                # Check if visualization window is closed
                viz_agent = self.agents.get('visualization')
                if viz_agent and viz_agent.quit_event.is_set():
                    print("\n🖥️  Visualization window closed")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring interrupted")
    
    def print_system_stats(self):
        """Print current system statistics"""
        print(f"\n📈 SYSTEM STATUS - {time.strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # Game Engine Stats
        game_engine = self.agents.get('game_engine')
        if game_engine:
            state = game_engine.get_public_state()
            print(f"🎮 Game Engine:")
            print(f"   Phase: {state.get('phase', 'unknown')}")
            print(f"   Round: {state.get('round_number', 0)}")
            print(f"   FPS: {state.get('fps', 0):.1f}")
            
            players = state.get('players', {})
            for pid, pdata in players.items():
                print(f"   Player {pid}: Score={pdata.get('score', 0)}, "
                      f"Foods={pdata.get('foods_eaten', 0)}, "
                      f"Deaths={pdata.get('deaths', 0)}")
        
        # Environment Stats
        env_agent = self.agents.get('environment')
        if env_agent:
            state = env_agent.get_public_state()
            print(f"🌍 Environment:")
            print(f"   Foods: {state.get('food_count', 0)}")
            print(f"   Obstacles: {state.get('obstacle_count', 0)}")
            print(f"   Food efficiency: {state.get('food_efficiency', 0):.1f}%")
        
        # Snake Logic Stats
        for snake_id in ['A', 'B']:
            snake_agent = self.agents.get(f'snake_{snake_id.lower()}')
            if snake_agent:
                state = snake_agent.get_public_state()
                print(f"🐍 Snake {snake_id}:")
                print(f"   Alive: {state.get('alive', False)}")
                print(f"   Length: {state.get('length', 0)}")
                print(f"   Moves: {state.get('moves_count', 0)}")
        
        # AI Training Stats
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                state = ai_agent.get_public_state()
                print(f"🤖 AI {snake_id}:")
                print(f"   Episode: {state.get('episode', 0)}")
                print(f"   Epsilon: {state.get('epsilon', 0):.3f}")
                print(f"   Avg Reward: {state.get('avg_reward', 0):.1f}")
                print(f"   Memory: {state.get('memory_size', 0)}")
        
        # Visualization Stats
        viz_agent = self.agents.get('visualization')
        if viz_agent:
            state = viz_agent.get_public_state()
            print(f"🎨 Visualization:")
            print(f"   FPS: {state.get('fps', 0):.1f}")
            print(f"   Effects: {state.get('effects_count', 0)}")
            print(f"   Display: {'Active' if state.get('display_initialized', False) else 'Inactive'}")
        
        print("-" * 50)
    
    def run_test_scenario(self):
        """Run a test scenario to demonstrate the system"""
        print("\n🧪 Running test scenario...")
        
        # Let the system run and stabilize
        time.sleep(3)
        
        # Test 1: Check agent communication
        print("📡 Testing agent communication...")
        message_bus = self.registry.message_bus
        recent_messages = message_bus.get_message_log(20)
        print(f"   Recent messages: {len(recent_messages)}")
        
        # Test 2: Verify AI decision making
        print("🧠 Testing AI decision making...")
        ai_agent_a = self.agents.get('ai_a')
        ai_agent_b = self.agents.get('ai_b')
        
        if ai_agent_a and ai_agent_b:
            state_a = ai_agent_a.get_public_state()
            state_b = ai_agent_b.get_public_state()
            print(f"   AI A active: {state_a.get('game_active', False)}")
            print(f"   AI B active: {state_b.get('game_active', False)}")
        
        # Test 3: Environment dynamics
        print("🌍 Testing environment dynamics...")
        env_agent = self.agents.get('environment')
        if env_agent:
            stats = env_agent.get_environment_stats()
            print(f"   Foods spawned: {stats['spawn_stats']['foods_spawned']}")
            print(f"   Foods eaten: {stats['spawn_stats']['foods_eaten']}")
        
        # Test 4: Game engine coordination
        print("🎮 Testing game engine coordination...")
        game_engine = self.agents.get('game_engine')
        if game_engine:
            state = game_engine.get_public_state()
            print(f"   Game phase: {state.get('phase', 'unknown')}")
            print(f"   Agents ready: {len(state.get('agents_ready', {}))}")
        
        print("✅ Test scenario completed")
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        if not self.running:
            return
            
        print("\n🔄 Shutting down system...")
        self.running = False
        
        # Stop all agents
        self.registry.stop_all_agents()
        
        # Save AI models
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                try:
                    ai_agent.save_model()
                    print(f"💾 Saved AI model for snake {snake_id}")
                except Exception as e:
                    print(f"⚠️  Could not save AI model for snake {snake_id}: {e}")
        
        print("✅ System shutdown complete")
    
    def print_final_report(self):
        """Print final system report"""
        print("\n" + "="*60)
        print("📋 FINAL SYSTEM REPORT")
        print("="*60)
        
        # Game statistics
        game_engine = self.agents.get('game_engine')
        if game_engine:
            stats = game_engine.get_stats_summary()
            print("🎮 Game Statistics:")
            print(f"   Total games: {stats.get('total_games', 0)}")
            print(f"   Current round: {stats.get('current_round', 0)}")
            print(f"   Average FPS: {stats.get('average_fps', 0):.1f}")
            
            players = stats.get('players', {})
            for pid, pdata in players.items():
                print(f"   Player {pid}: {pdata.get('total_score', 0)} points, "
                      f"{pdata.get('total_foods', 0)} foods, "
                      f"{pdata.get('total_deaths', 0)} deaths")
        
        # Environment statistics
        env_agent = self.agents.get('environment')
        if env_agent:
            stats = env_agent.get_environment_stats()
            print("\n🌍 Environment Statistics:")
            print(f"   Foods spawned: {stats['spawn_stats']['foods_spawned']}")
            print(f"   Foods eaten: {stats['spawn_stats']['foods_eaten']}")
            print(f"   Foods expired: {stats['spawn_stats']['foods_expired']}")
            print(f"   Efficiency: {stats['spawn_stats']['efficiency']:.2%}")
        
        # AI training statistics
        print("\n🤖 AI Training Statistics:")
        for snake_id in ['A', 'B']:
            ai_agent = self.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                state = ai_agent.get_public_state()
                print(f"   Snake {snake_id}:")
                print(f"     Episodes: {state.get('episode', 0)}")
                print(f"     Final epsilon: {state.get('epsilon', 0):.3f}")
                print(f"     Average reward: {state.get('avg_reward', 0):.1f}")
                print(f"     Training steps: {state.get('training_steps', 0)}")
                print(f"     Network type: {state.get('network_type', 'Unknown')}")
        
        print("="*60)

def main():
    """Main function to run the complete multi-agent Snake game"""
    print("🐍 Multi-Agent Snake Game System")
    print("=" * 40)
    
    # Create the game system
    game = MultiAgentSnakeGame()
    
    try:
        # Setup phase
        game.create_agents()
        game.register_agents()
        
        # Start the system
        game.start_system()
        
        # Run test scenario
        game.run_test_scenario()
        
        # Monitor the system (run for 1650 seconds for extended training)
        game.monitor_system(duration=1650)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Shutdown and report
        game.shutdown()
        game.print_final_report()

if __name__ == "__main__":
    main()