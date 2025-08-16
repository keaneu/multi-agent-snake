"""
Game Engine Agent - Core game loop, state management, and coordination
Handles the master game state, timing, scoring, and coordinates all other agents
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from multi_agent_framework import BaseAgent, MessageType, Message

class GamePhase(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    RESETTING = "resetting"

@dataclass
class GameConfig:
    """Game configuration settings"""
    grid_width: int = 20
    grid_height: int = 20
    target_fps: int = 10
    max_food_count: int = 3
    points_per_food: int = 10
    points_per_second: int = 1
    collision_penalty: int = -50
    max_game_duration: int = 300  # seconds
    wrap_walls: bool = False
    
@dataclass
class PlayerStats:
    """Statistics for a single player"""
    score: int = 0
    foods_eaten: int = 0
    time_alive: float = 0.0
    deaths: int = 0
    last_death_reason: str = ""
    alive: bool = True

@dataclass
class GameState:
    """Complete game state"""
    phase: GamePhase = GamePhase.INITIALIZING
    round_number: int = 0
    game_time: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    
    # Player states
    players: Dict[str, PlayerStats] = field(default_factory=lambda: {
        "A": PlayerStats(),
        "B": PlayerStats()
    })
    
    # Game metrics
    total_games: int = 0
    frame_count: int = 0
    actual_fps: float = 0.0
    
    # System health
    agents_ready: Dict[str, bool] = field(default_factory=dict)
    last_heartbeat: Dict[str, float] = field(default_factory=dict)

class GameEngineAgent(BaseAgent):
    """
    Master coordinator agent that manages the entire game system
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        super().__init__("game_engine")
        self.config = config or GameConfig()
        self.game_state = GameState()
        self.state = {}
        
        # Timing control
        self.frame_timer = 0.0
        self.frame_interval = 1.0 / self.config.target_fps
        self.fps_counter = []
        self.fps_window = 60  # frames to average FPS over
        
        # Round management
        self.round_start_time = 0.0
        self.round_end_scheduled = False
        
        # Agent coordination
        self.required_agents = {
            "snake_logic_A", "snake_logic_B", 
            "environment", "visualization",
            "ai_training_A", "ai_training_B"
        }
        self.agent_timeouts = {}
        self.initial_check_done = False
        
    def initialize(self):
        """Initialize the game engine"""
        print(f"🎮 Game Engine initializing...")
        
        # Subscribe to all relevant message types
        self.subscribe_to_messages([
            MessageType.COLLISION_EVENT,
            MessageType.FOOD_EATEN,
            MessageType.GAME_OVER,
            MessageType.RESET_GAME,
            MessageType.AI_DECISION,
            MessageType.ENVIRONMENT_UPDATE,
            MessageType.AGENT_READY
        ])
        
        # Initialize game state
        self.game_state.phase = GamePhase.INITIALIZING # Start in initializing phase
        self.game_state.last_update_time = time.time()
        
        # Broadcast initial configuration
        self.broadcast_config()
        
        print(f"✅ Game Engine initialized and waiting for agents...")
        
    def wait_for_agents(self, timeout: float = 10.0):
        """Wait for required agents to come online"""
        print("⏳ Waiting for agents to report ready status...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = sum(1 for agent_id in self.required_agents 
                            if self.game_state.agents_ready.get(agent_id, False))
            
            if ready_count == len(self.required_agents):
                print(f"✅ All {len(self.required_agents)} agents ready")
                break
                
            time.sleep(0.1)
        else:
            missing = [aid for aid in self.required_agents 
                      if not self.game_state.agents_ready.get(aid, False)]
            print(f"⚠️  Timeout waiting for agents: {missing}")
    
    def broadcast_config(self):
        """Broadcast game configuration to all agents"""
        config_data = {
            "grid_width": self.config.grid_width,
            "grid_height": self.config.grid_height,
            "target_fps": self.config.target_fps,
            "max_food_count": self.config.max_food_count,
            "wrap_walls": self.config.wrap_walls
        }
        self.broadcast_message(MessageType.GAME_STATE, {
            "type": "config",
            "config": config_data
        })
    
    def update(self):
        """Main game engine update loop"""
        current_time = time.time()
        
        # Control frame rate
        if current_time - self.frame_timer < self.frame_interval:
            return
            
        self.frame_timer = current_time
        dt = current_time - self.game_state.last_update_time
        self.game_state.last_update_time = current_time
        
        # Update FPS tracking
        self.update_fps_tracking(dt)
        
        # Update game based on current phase
        if self.game_state.phase == GamePhase.INITIALIZING:
            if not self.initial_check_done:
                self.wait_for_agents()
                self.initial_check_done = True
                self.game_state.phase = GamePhase.READY
                print(f"✅ Game Engine ready - Config: {self.config.grid_width}x{self.config.grid_height} @ {self.config.target_fps}fps")

        elif self.game_state.phase == GamePhase.RUNNING:
            self.update_running_game(dt)
        elif self.game_state.phase == GamePhase.GAME_OVER:
            self.update_game_over()
        elif self.game_state.phase == GamePhase.READY:
            self.start_new_round()
            
        # Broadcast game state to all agents
        self.broadcast_game_state()
        
        # Update internal state for framework
        self.state = self.get_public_state()
    
    def update_fps_tracking(self, dt: float):
        """Track actual FPS performance"""
        if dt > 0:
            fps = 1.0 / dt
            self.fps_counter.append(fps)
            
            if len(self.fps_counter) > self.fps_window:
                self.fps_counter.pop(0)
                
            self.game_state.actual_fps = sum(self.fps_counter) / len(self.fps_counter)
            self.game_state.frame_count += 1
    
    def update_running_game(self, dt: float):
        """Update game during running phase"""
        self.game_state.game_time += dt
        
        # Update player alive time
        for player_id, stats in self.game_state.players.items():
            if not self.is_player_dead(player_id):
                stats.time_alive += dt
                
                # Award points for staying alive
                if stats.time_alive >= 1.0:  # Every second
                    stats.score += self.config.points_per_second
                    stats.time_alive = 0.0
        
        # Check for game timeout
        if self.game_state.game_time > self.config.max_game_duration:
            self.end_game("timeout")
            
        # Check if all players are dead
        if self.all_players_dead():
            self.end_game("all_dead")
    
    def update_game_over(self):
        """Handle game over state"""
        if not self.round_end_scheduled:
            self.round_end_scheduled = True
            # Schedule reset after 2 seconds
            threading.Timer(2.0, self.reset_game).start()
    
    def start_new_round(self):
        """Start a new game round"""
        print(f"🚀 Starting round {self.game_state.round_number + 1}")
        
        self.game_state.phase = GamePhase.RUNNING
        self.game_state.round_number += 1
        self.game_state.game_time = 0.0
        self.round_start_time = time.time()
        self.round_end_scheduled = False
        
        # Reset player stats for this round
        for stats in self.game_state.players.values():
            stats.time_alive = 0.0
            
        # Request game reset from all agents
        self.broadcast_message(MessageType.RESET_GAME, {
            "round_number": self.game_state.round_number,
            "config": self.get_config_dict()
        })
    
    def end_game(self, reason: str):
        """End the current game"""
        if self.game_state.phase == GamePhase.RUNNING:
            print(f"🏁 Game ended: {reason}")
            
            self.game_state.phase = GamePhase.GAME_OVER
            self.game_state.total_games += 1
            
            # Determine winner
            winner = self.determine_winner()
            
            # Broadcast game over
            self.broadcast_message(MessageType.GAME_OVER, {
                "reason": reason,
                "winner": winner,
                "final_scores": {pid: stats.score for pid, stats in self.game_state.players.items()},
                "game_time": self.game_state.game_time,
                "round_number": self.game_state.round_number
            })
    
    def reset_game(self):
        """Reset game to ready state"""
        print("🔄 Resetting game...")
        
        self.game_state.phase = GamePhase.READY
        
        # Reset per-round player stats
        for player_id, stats in self.game_state.players.items():
            stats.score = 0
            stats.foods_eaten = 0
            stats.time_alive = 0.0
            stats.last_death_reason = ""
            stats.alive = True
    
    def process_message(self, message: Message):
        """Process incoming messages from other agents"""
        try:
            if message.type == MessageType.FOOD_EATEN:
                self.handle_food_eaten(message.data)
                
            elif message.type == MessageType.COLLISION_EVENT:
                self.handle_collision(message.data)
                
            elif message.type == MessageType.GAME_OVER:
                self.handle_game_over_request(message.data)
                
            elif message.type == MessageType.RESET_GAME:
                if message.sender != self.agent_id:  # Don't process our own resets
                    self.reset_game()
                    
            elif message.type == MessageType.AI_DECISION:
                self.handle_ai_decision(message.data)
                
            elif message.type == MessageType.ENVIRONMENT_UPDATE:
                self.handle_environment_update(message.data)
                
            elif message.type == MessageType.AGENT_READY:
                self.handle_agent_ready(message.sender)

            # Update agent heartbeat
            self.game_state.last_heartbeat[message.sender] = time.time()
            
        except Exception as e:
            print(f"❌ Error processing message from {message.sender}: {e}")
    
    def handle_agent_ready(self, agent_id: str):
        """Handle agent ready messages"""
        if agent_id in self.required_agents:
            self.game_state.agents_ready[agent_id] = True
            print(f"👍 Agent {agent_id} reported ready.")

    def handle_food_eaten(self, data: Dict[str, Any]):
        """Handle food eaten event"""
        player_id = data.get("player_id", "A")
        points = data.get("points", self.config.points_per_food)
        
        if player_id in self.game_state.players:
            self.game_state.players[player_id].score += points
            self.game_state.players[player_id].foods_eaten += 1
            
            print(f"🍎 Player {player_id} ate food (+{points} points)")
            
            # Broadcast score update
            self.broadcast_message(MessageType.SCORE_UPDATE, {
                "player_id": player_id,
                "new_score": self.game_state.players[player_id].score,
                "points_gained": points
            })
    
    def handle_collision(self, data: Dict[str, Any]):
        """Handle collision event"""
        player_id = data.get("player_id", "A")
        collision_type = data.get("collision_type", "unknown")
        
        if player_id in self.game_state.players:
            stats = self.game_state.players[player_id]
            if stats.alive: # Process only if player is alive
                stats.alive = False
                stats.deaths += 1
                stats.last_death_reason = collision_type
                stats.score += self.config.collision_penalty

                print(f"💥 Player {player_id} collision: {collision_type}")

                # Check if game should end
                if self.should_end_game_on_collision():
                    self.end_game(f"collision_{collision_type}")
    
    def handle_ai_decision(self, data: Dict[str, Any]):
        """Handle AI decision updates"""
        # Track AI performance metrics
        pass
    
    def handle_environment_update(self, data: Dict[str, Any]):
        """Handle environment state updates"""
        # Track environment changes
        pass
    
    def handle_game_over_request(self, data: Dict[str, Any]):
        """Handle game over requests from other agents"""
        reason = data.get("reason", "external_request")
        self.end_game(reason)
    
    def should_end_game_on_collision(self) -> bool:
        """Determine if game should end based on collision"""
        # End if all players are dead or only one alive
        alive_count = sum(1 for pid in self.game_state.players.keys() 
                         if not self.is_player_dead(pid))
        return alive_count <= 1
    
    def is_player_dead(self, player_id: str) -> bool:
        """Check if a player is dead"""
        if player_id in self.game_state.players:
            return not self.game_state.players[player_id].alive
        return True # Assume dead if not in player list
    
    def all_players_dead(self) -> bool:
        """Check if all players are dead"""
        return all(self.is_player_dead(pid) for pid in self.game_state.players.keys())
    
    def determine_winner(self) -> Optional[str]:
        """Determine the winner based on scores"""
        if not self.game_state.players:
            return None
            
        max_score = max(stats.score for stats in self.game_state.players.values())
        winners = [pid for pid, stats in self.game_state.players.items() 
                  if stats.score == max_score]
        
        return winners[0] if len(winners) == 1 else "tie"
    
    def broadcast_game_state(self):
        """Broadcast current game state to all agents"""
        self.broadcast_message(MessageType.GAME_STATE, {
            "type": "update",
            "phase": self.game_state.phase.value,
            "round_number": self.game_state.round_number,
            "game_time": self.game_state.game_time,
            "players": {pid: {
                "score": stats.score,
                "foods_eaten": stats.foods_eaten,
                "deaths": stats.deaths
            } for pid, stats in self.game_state.players.items()},
            "fps": self.game_state.actual_fps,
            "frame_count": self.game_state.frame_count
        })
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "grid_width": self.config.grid_width,
            "grid_height": self.config.grid_height,
            "target_fps": self.config.target_fps,
            "max_food_count": self.config.max_food_count,
            "points_per_food": self.config.points_per_food,
            "points_per_second": self.config.points_per_second,
            "collision_penalty": self.config.collision_penalty,
            "wrap_walls": self.config.wrap_walls
        }
    
    def get_public_state(self) -> Dict[str, Any]:
        """Get public state for other systems"""
        return {
            "phase": self.game_state.phase.value,
            "round_number": self.game_state.round_number,
            "game_time": self.game_state.game_time,
            "total_games": self.game_state.total_games,
            "fps": self.game_state.actual_fps,
            "players": {
                pid: {
                    "score": stats.score,
                    "foods_eaten": stats.foods_eaten,
                    "time_alive": stats.time_alive,
                    "deaths": stats.deaths,
                    "last_death_reason": stats.last_death_reason
                } for pid, stats in self.game_state.players.items()
            },
            "agents_ready": dict(self.game_state.agents_ready),
            "config": self.get_config_dict()
        }
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_games": self.game_state.total_games,
            "current_round": self.game_state.round_number,
            "average_fps": self.game_state.actual_fps,
            "total_frames": self.game_state.frame_count,
            "players": {
                pid: {
                    "total_score": stats.score,
                    "total_foods": stats.foods_eaten,
                    "total_deaths": stats.deaths
                } for pid, stats in self.game_state.players.items()
            }
        }

if __name__ == "__main__":
    # Test the game engine
    from multi_agent_framework import AgentRegistry
    
    config = GameConfig(
        grid_width=15,
        grid_height=15,
        target_fps=5,  # Slower for testing
        max_food_count=2
    )
    
    registry = AgentRegistry()
    game_engine = GameEngineAgent(config)
    
    registry.register_agent(game_engine)
    registry.start_all_agents()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
        
        # Test food eaten event
        game_engine.send_message(
            MessageType.FOOD_EATEN,
            "game_engine",
            {"player_id": "A", "points": 15}
        )
        
        time.sleep(5)
        
        # Check stats
        print("\n📊 Game Stats:")
        stats = game_engine.get_stats_summary()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    finally:
        registry.stop_all_agents()