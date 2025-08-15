"""
Snake Logic Agent - Movement, collision detection, and growth mechanics
Handles individual snake behavior, physics, and interactions
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from multi_agent_framework import BaseAgent, MessageType, Message

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    
    def opposite(self):
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }[self]

class CollisionType(Enum):
    WALL = "wall"
    SELF = "self"
    OTHER_SNAKE = "other_snake"
    OBSTACLE = "obstacle"

@dataclass
class Position:
    x: int
    y: int
    
    def __add__(self, other):
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple):
            return Position(self.x + other[0], self.y + other[1])
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1]
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def to_tuple(self):
        return (self.x, self.y)

@dataclass
class SnakeSegment:
    position: Position
    age: int = 0  # How long this segment has existed
    
@dataclass
class PowerUp:
    type: str
    duration: float
    strength: float = 1.0
    
@dataclass
class SnakeState:
    """Complete state of a single snake"""
    snake_id: str
    segments: List[SnakeSegment] = field(default_factory=list)
    direction: Direction = Direction.RIGHT
    next_direction: Direction = Direction.RIGHT
    
    # Growth mechanics
    growth_pending: int = 0
    max_length: int = 100
    
    # Movement
    speed_multiplier: float = 1.0
    last_move_time: float = 0.0
    
    # Status
    alive: bool = True
    death_reason: str = ""
    death_position: Optional[Position] = None
    
    # Power-ups
    active_powers: Dict[str, PowerUp] = field(default_factory=dict)
    
    # Statistics
    moves_count: int = 0
    foods_eaten: int = 0
    time_alive: float = 0.0
    
    # Physics
    momentum: float = 0.0
    last_direction_change: float = 0.0

class SnakeLogicAgent(BaseAgent):
    """
    Handles all snake logic including movement, collision detection, and growth
    Can manage multiple snakes or focus on a single snake
    """
    
    def __init__(self, snake_id: str):
        super().__init__(f"snake_logic_{snake_id}")
        self.snake_id = snake_id
        self.snake_state = SnakeState(snake_id)
        
        # Game configuration (received from game engine)
        self.grid_width = 20
        self.grid_height = 20
        self.wrap_walls = False
        self.move_interval = 0.1  # seconds between moves
        
        # World state (received from environment agent)
        self.food_positions: List[Position] = []
        self.obstacles: List[Position] = []
        self.other_snakes: Dict[str, List[Position]] = {}
        
        # Movement queue for smooth control
        self.direction_queue: List[Direction] = []
        self.max_queue_size = 3
        
    def initialize(self):
        """Initialize the snake logic agent"""
        print(f"🐍 Snake Logic Agent {self.snake_id} initializing...")
        
        # Subscribe to relevant messages
        self.subscribe_to_messages([
            MessageType.GAME_STATE,
            MessageType.MOVE_COMMAND,
            MessageType.ENVIRONMENT_UPDATE,
            MessageType.RESET_GAME
        ])
        
        # Initialize snake in default position
        self.reset_snake()
        
        # Update internal state
        self.state = self.get_public_state()
        
        print(f"✅ Snake Logic Agent {self.snake_id} ready")
    
    def reset_snake(self):
        """Reset snake to initial state"""
        # Default spawn position (will be overridden by game engine)
        start_x = 3 if self.snake_id == "A" else self.grid_width - 4
        start_y = self.grid_height // 2
        
        # Create initial snake with 3 segments
        self.snake_state.segments = [
            SnakeSegment(Position(start_x, start_y)),
            SnakeSegment(Position(start_x - 1, start_y)),
            SnakeSegment(Position(start_x - 2, start_y))
        ]
        
        self.snake_state.direction = Direction.RIGHT if self.snake_id == "A" else Direction.LEFT
        self.snake_state.next_direction = self.snake_state.direction
        self.snake_state.alive = True
        self.snake_state.death_reason = ""
        self.snake_state.growth_pending = 0
        self.snake_state.moves_count = 0
        self.snake_state.foods_eaten = 0
        self.snake_state.time_alive = 0.0
        self.snake_state.last_move_time = time.time()
        self.snake_state.active_powers.clear()
        self.direction_queue.clear()
        
        print(f"🔄 Snake {self.snake_id} reset at {start_x}, {start_y}")
    
    def update(self):
        """Main update loop for snake logic"""
        if not self.snake_state.alive:
            return
            
        current_time = time.time()
        dt = current_time - self.snake_state.last_move_time
        
        # Update power-ups
        self.update_power_ups(dt)
        
        # Update alive time
        self.snake_state.time_alive += dt
        
        # Check if it's time to move
        move_interval = self.move_interval / self.snake_state.speed_multiplier
        if dt >= move_interval:
            self.move_snake()
            self.snake_state.last_move_time = current_time
        
        # Update state for other agents
        self.state = self.get_public_state()
    
    def move_snake(self):
        """Execute snake movement"""
        if not self.snake_state.alive:
            return
            
        # Process direction queue
        self.process_direction_queue()
        
        # Calculate new head position
        head = self.snake_state.segments[0]
        direction_vector = self.snake_state.direction.value
        new_head_pos = head.position + direction_vector
        
        # Handle wall wrapping/collision
        new_head_pos = self.handle_wall_interaction(new_head_pos)
        if new_head_pos is None:  # Wall collision
            return
        
        # Check for collisions before moving
        collision_type = self.check_collisions(new_head_pos)
        if collision_type:
            self.handle_collision(collision_type, new_head_pos)
            return
        
        # Create new head segment
        new_head = SnakeSegment(new_head_pos)
        self.snake_state.segments.insert(0, new_head)
        
        # Handle growth or remove tail
        if self.snake_state.growth_pending > 0:
            self.snake_state.growth_pending -= 1
        else:
            # Remove tail segment
            self.snake_state.segments.pop()
        
        # Age all segments
        for segment in self.snake_state.segments:
            segment.age += 1
        
        # Update statistics
        self.snake_state.moves_count += 1
        
        # Check for food consumption
        self.check_food_consumption(new_head_pos)
        
        # Broadcast movement update
        self.broadcast_snake_update()
    
    def process_direction_queue(self):
        """Process queued direction changes"""
        if self.direction_queue:
            next_dir = self.direction_queue.pop(0)
            
            # Prevent immediate reversal (can't go directly backwards)
            if next_dir != self.snake_state.direction.opposite():
                self.snake_state.direction = next_dir
                self.snake_state.next_direction = next_dir
                self.snake_state.last_direction_change = time.time()
    
    def handle_wall_interaction(self, position: Position) -> Optional[Position]:
        """Handle wall collision or wrapping"""
        if self.wrap_walls:
            # Wrap around edges
            new_x = position.x % self.grid_width
            new_y = position.y % self.grid_height
            return Position(new_x, new_y)
        else:
            # Check bounds
            if (position.x < 0 or position.x >= self.grid_width or 
                position.y < 0 or position.y >= self.grid_height):
                self.handle_collision(CollisionType.WALL, position)
                return None
            return position
    
    def check_collisions(self, position: Position) -> Optional[CollisionType]:
        """Check for collisions at given position"""
        # Self collision (skip head)
        for segment in self.snake_state.segments[1:]:
            if segment.position == position:
                return CollisionType.SELF
        
        # Other snake collision
        for other_id, other_segments in self.other_snakes.items():
            if other_id != self.snake_id:
                for other_pos in other_segments:
                    if Position(other_pos[0], other_pos[1]) == position:
                        return CollisionType.OTHER_SNAKE
        
        # Obstacle collision
        if position in self.obstacles:
            return CollisionType.OBSTACLE
        
        return None
    
    def handle_collision(self, collision_type: CollisionType, position: Position):
        """Handle collision event"""
        self.snake_state.alive = False
        self.snake_state.death_reason = collision_type.value
        self.snake_state.death_position = position
        
        print(f"💥 Snake {self.snake_id} died: {collision_type.value} at {position.x}, {position.y}")
        
        # Broadcast collision event
        self.send_message(MessageType.COLLISION_EVENT, "game_engine", {
            "player_id": self.snake_id,
            "collision_type": collision_type.value,
            "position": position.to_tuple(),
            "time_alive": self.snake_state.time_alive,
            "moves_count": self.snake_state.moves_count
        })
    
    def check_food_consumption(self, position: Position):
        """Check if snake ate food at position"""
        for food_pos in self.food_positions:
            if food_pos == position:
                self.eat_food(food_pos)
                break
    
    def eat_food(self, food_position: Position):
        """Handle food consumption"""
        self.snake_state.growth_pending += 2  # Grow by 2 segments
        self.snake_state.foods_eaten += 1
        
        print(f"🍎 Snake {self.snake_id} ate food at {food_position.x}, {food_position.y}")
        
        # Broadcast food eaten event
        self.send_message(MessageType.FOOD_EATEN, "game_engine", {
            "player_id": self.snake_id,
            "position": food_position.to_tuple(),
            "growth_amount": 2,
            "total_foods": self.snake_state.foods_eaten
        })
        
        # Remove food from local tracking
        self.food_positions = [pos for pos in self.food_positions if pos != food_position]
    
    def update_power_ups(self, dt: float):
        """Update active power-ups"""
        expired_powers = []
        
        for power_name, power in self.snake_state.active_powers.items():
            power.duration -= dt
            if power.duration <= 0:
                expired_powers.append(power_name)
        
        # Remove expired power-ups
        for power_name in expired_powers:
            self.remove_power_up(power_name)
    
    def apply_power_up(self, power_type: str, duration: float, strength: float = 1.0):
        """Apply a power-up to the snake"""
        power = PowerUp(power_type, duration, strength)
        self.snake_state.active_powers[power_type] = power
        
        # Apply immediate effects
        if power_type == "speed":
            self.snake_state.speed_multiplier = 1.0 + strength
        elif power_type == "slow":
            self.snake_state.speed_multiplier = max(0.1, 1.0 - strength)
        elif power_type == "growth":
            self.snake_state.growth_pending += int(strength)
        
        print(f"⚡ Snake {self.snake_id} gained power: {power_type}")
    
    def remove_power_up(self, power_type: str):
        """Remove a power-up from the snake"""
        if power_type in self.snake_state.active_powers:
            del self.snake_state.active_powers[power_type]
            
            # Remove effects
            if power_type in ["speed", "slow"]:
                self.snake_state.speed_multiplier = 1.0
            
            print(f"⏰ Snake {self.snake_id} lost power: {power_type}")
    
    def queue_direction_change(self, direction: Direction):
        """Queue a direction change (for smooth input handling)"""
        if len(self.direction_queue) < self.max_queue_size:
            # Don't queue the same direction twice
            if not self.direction_queue or self.direction_queue[-1] != direction:
                self.direction_queue.append(direction)
                return True
        # Return False but don't spam warnings - this is normal during high-frequency AI decisions
        return False
    
    def get_head_position(self) -> Position:
        """Get current head position"""
        return self.snake_state.segments[0].position if self.snake_state.segments else Position(0, 0)
    
    def get_length(self) -> int:
        """Get current snake length"""
        return len(self.snake_state.segments)
    
    def get_predicted_head_position(self, steps: int = 1) -> Position:
        """Predict where the head will be after N steps"""
        head = self.get_head_position()
        direction_vector = self.snake_state.direction.value
        
        predicted_pos = Position(
            head.x + direction_vector[0] * steps,
            head.y + direction_vector[1] * steps
        )
        
        if self.wrap_walls:
            predicted_pos.x = predicted_pos.x % self.grid_width
            predicted_pos.y = predicted_pos.y % self.grid_height
        
        return predicted_pos
    
    def can_move_to(self, position: Position) -> bool:
        """Check if snake can safely move to position"""
        return self.check_collisions(position) is None
    
    def get_safe_directions(self) -> List[Direction]:
        """Get list of safe directions to move"""
        safe_directions = []
        head = self.get_head_position()
        
        for direction in Direction:
            # Don't allow immediate reversal
            if direction == self.snake_state.direction.opposite():
                continue
                
            next_pos = head + direction.value
            next_pos = self.handle_wall_interaction(next_pos)
            
            if next_pos and self.can_move_to(next_pos):
                safe_directions.append(direction)
        
        return safe_directions
    
    def process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.GAME_STATE:
                self.handle_game_state_update(message.data)
                
            elif message.type == MessageType.MOVE_COMMAND:
                self.handle_move_command(message.data)
                
            elif message.type == MessageType.ENVIRONMENT_UPDATE:
                self.handle_environment_update(message.data)
                
            elif message.type == MessageType.RESET_GAME:
                self.handle_game_reset(message.data)
                
        except Exception as e:
            print(f"❌ Snake {self.snake_id} error processing message: {e}")
    
    def handle_game_state_update(self, data: Dict[str, Any]):
        """Handle game state updates from game engine"""
        if data.get("type") == "config":
            config = data.get("config", {})
            self.grid_width = config.get("grid_width", self.grid_width)
            self.grid_height = config.get("grid_height", self.grid_height)
            self.wrap_walls = config.get("wrap_walls", self.wrap_walls)
            
            # Calculate move interval from target FPS
            target_fps = config.get("target_fps", 10)
            self.move_interval = 1.0 / target_fps
            
            print(f"📝 Snake {self.snake_id} updated config: {self.grid_width}x{self.grid_height}")
    
    def handle_move_command(self, data: Dict[str, Any]):
        """Handle movement commands (from AI or user input)"""
        target_snake = data.get("snake_id")
        if target_snake != self.snake_id:
            return
            
        direction_str = data.get("direction", "").upper()
        try:
            direction = Direction[direction_str]
            success = self.queue_direction_change(direction)
            # Removed spam warning - queue full is normal during high-frequency AI decisions
        except KeyError:
            print(f"❌ Invalid direction: {direction_str}")
    
    def handle_environment_update(self, data: Dict[str, Any]):
        """Handle environment updates"""
        # Update food positions
        foods = data.get("foods", [])
        self.food_positions = [Position(x, y) for x, y in foods]
        
        # Update obstacles
        obstacles = data.get("obstacles", [])
        self.obstacles = [Position(x, y) for x, y in obstacles]
        
        # Update other snakes
        snakes = data.get("snakes", {})
        self.other_snakes = {
            snake_id: segments for snake_id, segments in snakes.items()
            if snake_id != self.snake_id
        }
    
    def handle_game_reset(self, data: Dict[str, Any]):
        """Handle game reset"""
        # Update configuration if provided
        config = data.get("config", {})
        if config:
            self.grid_width = config.get("grid_width", self.grid_width)
            self.grid_height = config.get("grid_height", self.grid_height)
            self.wrap_walls = config.get("wrap_walls", self.wrap_walls)
        
        # Reset snake state
        self.reset_snake()
        
        # Broadcast that we're ready
        self.broadcast_snake_update()
    
    def broadcast_snake_update(self):
        """Broadcast current snake state to other agents"""
        self.broadcast_message(MessageType.GAME_STATE, {
            "type": "snake_update",
            "snake_id": self.snake_id,
            "segments": [seg.position.to_tuple() for seg in self.snake_state.segments],
            "direction": self.snake_state.direction.name,
            "alive": self.snake_state.alive,
            "length": len(self.snake_state.segments),
            "growth_pending": self.snake_state.growth_pending
        })
    
    def get_public_state(self) -> Dict[str, Any]:
        """Get public state for other systems"""
        return {
            "snake_id": self.snake_id,
            "alive": self.snake_state.alive,
            "segments": [seg.position.to_tuple() for seg in self.snake_state.segments],
            "head_position": self.get_head_position().to_tuple(),
            "direction": self.snake_state.direction.name,
            "length": len(self.snake_state.segments),
            "growth_pending": self.snake_state.growth_pending,
            "foods_eaten": self.snake_state.foods_eaten,
            "moves_count": self.snake_state.moves_count,
            "time_alive": self.snake_state.time_alive,
            "speed_multiplier": self.snake_state.speed_multiplier,
            "active_powers": list(self.snake_state.active_powers.keys()),
            "death_reason": self.snake_state.death_reason,
            "safe_directions": [d.name for d in self.get_safe_directions()]
        }

if __name__ == "__main__":
    # Test the snake logic agent
    from multi_agent_framework import AgentRegistry
    import threading
    
    registry = AgentRegistry()
    
    # Create two snake agents
    snake_a = SnakeLogicAgent("A")
    snake_b = SnakeLogicAgent("B")
    
    registry.register_agent(snake_a)
    registry.register_agent(snake_b)
    
    registry.start_all_agents()
    
    try:
        time.sleep(2)
        
        # Test movement commands
        print("\n🎮 Testing movement commands...")
        
        snake_a.send_message(MessageType.MOVE_COMMAND, f"snake_logic_A", {
            "snake_id": "A",
            "direction": "UP"
        })
        
        snake_b.send_message(MessageType.MOVE_COMMAND, f"snake_logic_B", {
            "snake_id": "B", 
            "direction": "DOWN"
        })
        
        time.sleep(3)
        
        # Check states
        print(f"\n📊 Snake A state: {snake_a.get_public_state()}")
        print(f"📊 Snake B state: {snake_b.get_public_state()}")
        
    finally:
        registry.stop_all_agents()