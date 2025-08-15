"""
Test Multi-Agent Snake System - Console version without pygame
Tests all 6 agents working together in a simplified environment
"""

import time
import signal
import sys
from multi_agent_framework import AgentRegistry
from game_engine_agent import GameEngineAgent, GameConfig
from snake_logic_agent import SnakeLogicAgent
from environment_agent import EnvironmentAgent, EnvironmentConfig
from ai_training_agent import AITrainingAgent, TrainingConfig

class ConsoleVisualizationAgent:
    """Simplified console-based visualization for testing"""
    
    def __init__(self):
        self.agent_id = "visualization"
        self.grid_width = 20
        self.grid_height = 20
        self.snakes = {}
        self.foods = []
        self.obstacles = []
        self.frame_count = 0
        
    def set_message_bus(self, message_bus):
        self.message_bus = message_bus
        
    def start(self):
        print("🎨 Console Visualization Agent started")
        
    def stop(self):
        print("🎨 Console Visualization Agent stopped")
        
    def get_state(self):
        return {"frame_count": self.frame_count}
    
    def print_game_state(self):
        """Print a simple ASCII representation of the game"""
        if not self.snakes:
            return
            
        # Create grid
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Add obstacles
        for obs_x, obs_y in self.obstacles:
            if 0 <= obs_x < self.grid_width and 0 <= obs_y < self.grid_height:
                grid[obs_y][obs_x] = '#'
        
        # Add foods
        for food_x, food_y in self.foods:
            if 0 <= food_x < self.grid_width and 0 <= food_y < self.grid_height:
                grid[food_y][food_x] = '*'
        
        # Add snakes
        for snake_id, segments in self.snakes.items():
            symbol = 'A' if snake_id == 'A' else 'B'
            for i, (x, y) in enumerate(segments):
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    if i == 0:  # Head
                        grid[y][x] = symbol
                    else:  # Body
                        grid[y][x] = symbol.lower()
        
        # Print grid
        print("\n" + "="*50)
        for row in grid:
            print(' '.join(row))
        print("="*50)

def test_multi_agent_system():
    """Test the complete multi-agent system without pygame"""
    print("🐍 Testing Multi-Agent Snake System")
    print("=" * 50)
    
    # Create registry
    registry = AgentRegistry()
    
    # Create agents
    print("🔧 Creating agents...")
    
    # 1. Game Engine
    game_config = GameConfig(
        grid_width=15,
        grid_height=10,
        target_fps=2,  # Very slow for testing
        max_food_count=2,
        points_per_food=10,
        wrap_walls=False
    )
    game_engine = GameEngineAgent(game_config)
    registry.register_agent(game_engine)
    
    # 2. Snake Logic Agents
    snake_a = SnakeLogicAgent("A")
    snake_b = SnakeLogicAgent("B")
    registry.register_agent(snake_a)
    registry.register_agent(snake_b)
    
    # 3. Environment Agent
    env_config = EnvironmentConfig(
        grid_width=15,
        grid_height=10,
        max_food_count=2,
        food_spawn_rate=0.5,
        obstacle_count=2,
        dynamic_obstacles=False
    )
    environment = EnvironmentAgent(env_config)
    registry.register_agent(environment)
    
    # 4. Console Visualization (simplified)
    console_viz = ConsoleVisualizationAgent()
    
    # 5. AI Training Agents
    ai_config = TrainingConfig(
        hidden_size=64,
        learning_rate=0.01,
        epsilon_start=0.8,
        epsilon_decay=0.98,
        memory_size=1000,
        batch_size=16
    )
    
    ai_a = AITrainingAgent("A", ai_config)
    ai_b = AITrainingAgent("B", ai_config)
    registry.register_agent(ai_a)
    registry.register_agent(ai_b)
    
    print(f"✅ Created {len(registry.agents)} agents")
    
    # Start system
    print("\n🚀 Starting system...")
    registry.start_all_agents()
    
    try:
        # Let system initialize
        time.sleep(3)
        
        print("\n📊 Testing agent communication...")
        
        # Test 1: Check message flow
        message_log = registry.message_bus.get_message_log(10)
        print(f"Recent messages: {len(message_log)}")
        for msg in message_log[-3:]:
            print(f"  {msg['sender']} -> {msg['recipient']}: {msg['type']}")
        
        # Test 2: Monitor for 20 seconds
        print(f"\n🔍 Monitoring system for 20 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 20:
            # Update console visualization
            messages = registry.message_bus.get_message_log(5)
            for msg in messages:
                if msg['type'] == 'game_state' and msg['data'].get('type') == 'snake_update':
                    snake_id = msg['data'].get('snake_id')
                    segments = msg['data'].get('segments', [])
                    if snake_id:
                        console_viz.snakes[snake_id] = segments
                
                elif msg['type'] == 'environment_update':
                    console_viz.foods = msg['data'].get('foods', [])
                    console_viz.obstacles = msg['data'].get('obstacles', [])
            
            # Print stats every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"\n⏱️  Time: {int(elapsed)}s")
                
                # Game engine stats
                game_state = game_engine.get_public_state()
                print(f"🎮 Game: Phase={game_state.get('phase', 'unknown')}, "
                      f"Round={game_state.get('round_number', 0)}, "
                      f"FPS={game_state.get('fps', 0):.1f}")
                
                # Player scores
                players = game_state.get('players', {})
                for pid, pdata in players.items():
                    print(f"   Player {pid}: Score={pdata.get('score', 0)}, "
                          f"Alive={pdata.get('time_alive', 0):.1f}s")
                
                # Environment stats
                env_state = environment.get_public_state()
                print(f"🌍 Environment: Foods={env_state.get('food_count', 0)}, "
                      f"Obstacles={env_state.get('obstacle_count', 0)}")
                
                # Snake states
                for snake_id in ['A', 'B']:
                    snake_agent = registry.get_agent(f'snake_logic_{snake_id}')
                    if snake_agent:
                        snake_state = snake_agent.get_public_state()
                        print(f"🐍 Snake {snake_id}: Alive={snake_state.get('alive', False)}, "
                              f"Length={snake_state.get('length', 0)}, "
                              f"Moves={snake_state.get('moves_count', 0)}")
                
                # AI stats
                for snake_id in ['A', 'B']:
                    ai_agent = registry.get_agent(f'ai_training_{snake_id}')
                    if ai_agent:
                        ai_state = ai_agent.get_public_state()
                        print(f"🤖 AI {snake_id}: Episode={ai_state.get('episode', 0)}, "
                              f"Epsilon={ai_state.get('epsilon', 0):.3f}, "
                              f"Memory={ai_state.get('memory_size', 0)}")
                
                # Show game board
                console_viz.print_game_state()
            
            time.sleep(1)
        
        # Final statistics
        print(f"\n📋 FINAL RESULTS")
        print("=" * 50)
        
        # Game statistics
        final_game_state = game_engine.get_stats_summary()
        print(f"🎮 Game Results:")
        print(f"   Total games: {final_game_state.get('total_games', 0)}")
        print(f"   Rounds played: {final_game_state.get('current_round', 0)}")
        
        players = final_game_state.get('players', {})
        for pid, pdata in players.items():
            print(f"   Player {pid}: {pdata.get('total_score', 0)} points, "
                  f"{pdata.get('total_foods', 0)} foods eaten")
        
        # Environment statistics
        env_stats = environment.get_environment_stats()
        print(f"\n🌍 Environment Results:")
        print(f"   Foods spawned: {env_stats['spawn_stats']['foods_spawned']}")
        print(f"   Foods eaten: {env_stats['spawn_stats']['foods_eaten']}")
        print(f"   Food efficiency: {env_stats['spawn_stats']['efficiency']:.2%}")
        
        # AI learning progress
        print(f"\n🤖 AI Learning Results:")
        for snake_id in ['A', 'B']:
            ai_agent = registry.get_agent(f'ai_training_{snake_id}')
            if ai_agent:
                ai_state = ai_agent.get_public_state()
                print(f"   AI {snake_id}:")
                print(f"     Episodes completed: {ai_state.get('episode', 0)}")
                print(f"     Exploration rate: {ai_state.get('epsilon', 0):.3f}")
                print(f"     Average reward: {ai_state.get('avg_reward', 0):.1f}")
                print(f"     Network type: {ai_state.get('network_type', 'Unknown')}")
        
        # Communication statistics
        total_messages = len(registry.message_bus.message_log)
        print(f"\n📡 Communication Results:")
        print(f"   Total messages: {total_messages}")
        print(f"   Messages per second: {total_messages / 20:.1f}")
        
        # Agent health check
        print(f"\n✅ Agent Health Check:")
        for agent_id, agent in registry.agents.items():
            is_running = hasattr(agent, 'running') and getattr(agent, 'running', False)
            print(f"   {agent_id}: {'Running' if is_running else 'Stopped'}")
        
        print("\n🎉 Multi-agent system test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🔄 Shutting down system...")
        registry.stop_all_agents()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    test_multi_agent_system()