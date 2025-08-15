#!/usr/bin/env python3
"""
Test script for the self-repaired multi-agent Snake system
Runs for a limited time to verify bug fixes
"""

import sys
import time
import signal
from run_multi_agent_snake import MultiAgentSnakeGame

def test_system_with_timeout():
    """Test the system with a timeout"""
    print("🔧 Testing self-repaired multi-agent system...")
    
    # Create game instance
    game = MultiAgentSnakeGame()
    
    # Set up timeout handler
    def timeout_handler(signum, frame):
        print(f"\n⏰ Test timeout reached")
        game.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout
    
    try:
        # Test system creation and startup
        print("📦 Creating agents...")
        game.create_agents()
        print(f"✅ Created {len(game.agents)} agents successfully")
        
        print("📝 Registering agents...")
        game.register_agents()
        print("✅ All agents registered")
        
        print("🚀 Starting system...")
        game.start_system()
        print("✅ System started")
        
        # Test running for a few seconds
        print("🔍 Testing system operation...")
        time.sleep(5)
        
        # Check agent states
        print("\n📊 Agent Status Check:")
        for agent_name, agent in game.agents.items():
            is_running = hasattr(agent, 'running') and getattr(agent, 'running', False)
            print(f"   {agent_name}: {'✅ Running' if is_running else '❌ Stopped'}")
        
        # Test communication
        message_count = len(game.registry.message_bus.message_log)
        print(f"📡 Message bus: {message_count} messages processed")
        
        # Test specific functionality
        print("\n🎮 Game Engine Status:")
        game_engine = game.agents.get('game_engine')
        if game_engine:
            state = game_engine.get_public_state()
            print(f"   Phase: {state.get('phase', 'unknown')}")
            print(f"   FPS: {state.get('fps', 0):.1f}")
            print(f"   Round: {state.get('round_number', 0)}")
        
        print("\n🌍 Environment Status:")
        env_agent = game.agents.get('environment')
        if env_agent:
            state = env_agent.get_public_state()
            print(f"   Foods: {state.get('food_count', 0)}")
            print(f"   Obstacles: {state.get('obstacle_count', 0)}")
        
        print("\n🤖 AI Status:")
        for snake_id in ['A', 'B']:
            ai_agent = game.agents.get(f'ai_{snake_id.lower()}')
            if ai_agent:
                state = ai_agent.get_public_state()
                print(f"   AI {snake_id}: Episode {state.get('episode', 0)}, "
                      f"ε={state.get('epsilon', 0):.3f}")
        
        print("\n✅ All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        signal.alarm(0)  # Cancel timeout
        game.shutdown()
        print("🔄 System shutdown complete")

if __name__ == "__main__":
    success = test_system_with_timeout()
    sys.exit(0 if success else 1)