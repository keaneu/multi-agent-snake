"""
Environment Agent - Food spawning, boundaries, and game world logic
Manages the game world including food placement, obstacles, boundaries, and environmental effects
"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from multi_agent_framework import BaseAgent, MessageType, Message

class FoodType(Enum):
    NORMAL = "normal"
    SPEED_BOOST = "speed_boost"
    GROWTH_BOOST = "growth_boost"
    POINTS_MULTIPLIER = "points_multiplier"
    SHRINK = "shrink"

class ObstacleType(Enum):
    STATIC = "static"
    MOVING = "moving"
    DESTRUCTIBLE = "destructible"
    TELEPORTER = "teleporter"

@dataclass
class Position:
    x: int
    y: int
    
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
    
    def distance_to(self, other) -> float:
        if isinstance(other, Position):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        elif isinstance(other, tuple):
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)
        return float('inf')

@dataclass
class Food:
    position: Position
    food_type: FoodType
    points_value: int
    spawn_time: float
    duration: float = 30.0  # Food expires after 30 seconds
    special_data: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: float) -> bool:
        return current_time - self.spawn_time > self.duration
    
    def get_age(self, current_time: float) -> float:
        return current_time - self.spawn_time

@dataclass
class Obstacle:
    position: Position
    obstacle_type: ObstacleType
    size: Tuple[int, int] = (1, 1)  # width, height
    health: int = 1
    spawn_time: float = 0.0
    movement_pattern: Optional[str] = None
    special_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvironmentConfig:
    """Configuration for environment generation"""
    grid_width: int = 20
    grid_height: int = 20
    
    # Food settings
    max_food_count: int = 3
    food_spawn_rate: float = 2.0  # foods per second
    food_min_distance: int = 2    # minimum distance between foods
    special_food_chance: float = 0.3  # 30% chance for special food
    
    # Obstacle settings
    obstacle_count: int = 5
    obstacle_density: float = 0.1  # percentage of grid covered
    moving_obstacle_chance: float = 0.2
    
    # Boundary settings
    wrap_walls: bool = False
    safe_zone_size: int = 3  # safe area around spawn points
    
    # Environmental effects
    dynamic_obstacles: bool = True
    food_clustering: bool = False
    danger_zones: bool = False

class EnvironmentAgent(BaseAgent):
    """
    Manages the game world environment including food, obstacles, and boundaries
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__("environment")
        self.config = config or EnvironmentConfig()
        
        # World state
        self.foods: Dict[Position, Food] = {}
        self.obstacles: Dict[Position, Obstacle] = {}
        self.danger_zones: List[Tuple[Position, int]] = []  # position, radius
        
        # Snake tracking (for collision avoidance)
        self.snakes: Dict[str, List[Position]] = {}
        self.snake_spawn_points: Dict[str, Position] = {}
        
        # Timing
        self.last_food_spawn = 0.0
        self.last_obstacle_update = 0.0
        self.last_environment_broadcast = 0.0
        
        # Statistics
        self.total_foods_spawned = 0
        self.total_foods_eaten = 0
        self.total_foods_expired = 0
        
        # Generation parameters
        self.food_spawn_zones: List[Tuple[Position, int]] = []  # position, radius
        self.forbidden_zones: Set[Position] = set()
        
    def initialize(self):
        """Initialize the environment agent"""
        print("🌍 Environment Agent initializing...")
        
        # Subscribe to relevant messages
        self.subscribe_to_messages([
            MessageType.GAME_STATE,
            MessageType.FOOD_EATEN,
            MessageType.COLLISION_EVENT,
            MessageType.RESET_GAME
        ])
        
        # Generate initial world
        self.generate_world()
        
        # Initialize spawn zones
        self.setup_spawn_zones()
        
        # Update state
        self.state = self.get_public_state()
        
        print(f"✅ Environment Agent ready - {self.config.grid_width}x{self.config.grid_height} world")
    
    def setup_spawn_zones(self):
        """Setup food spawn zones and forbidden areas"""
        center_x = self.config.grid_width // 2
        center_y = self.config.grid_height // 2
        
        if self.config.food_clustering:
            # Create clustered spawn zones
            self.food_spawn_zones = [
                (Position(center_x, center_y), 3),
                (Position(center_x - 6, center_y - 4), 2),
                (Position(center_x + 6, center_y + 4), 2),
            ]
        else:
            # Whole map is spawn zone
            self.food_spawn_zones = [
                (Position(center_x, center_y), max(self.config.grid_width, self.config.grid_height))
            ]
        
        # Mark safe zones around snake spawns as forbidden for obstacles
        for spawn_pos in self.snake_spawn_points.values():
            for dx in range(-self.config.safe_zone_size, self.config.safe_zone_size + 1):
                for dy in range(-self.config.safe_zone_size, self.config.safe_zone_size + 1):
                    pos = Position(spawn_pos.x + dx, spawn_pos.y + dy)
                    if self.is_valid_position(pos):
                        self.forbidden_zones.add(pos)
    
    def generate_world(self):
        """Generate the initial world layout"""
        self.clear_world()
        
        # Set default snake spawn points
        self.snake_spawn_points = {
            "A": Position(3, self.config.grid_height // 2),
            "B": Position(self.config.grid_width - 4, self.config.grid_height // 2)
        }
        
        # Generate initial obstacles
        self.generate_obstacles()
        
        # Generate initial foods
        for _ in range(self.config.max_food_count):
            self.spawn_food()
        
        # Generate danger zones if enabled
        if self.config.danger_zones:
            self.generate_danger_zones()
        
        # Broadcast initial environment
        self.broadcast_environment_update()
        # Announce readiness
        self.send_message(MessageType.AGENT_READY, "game_engine", {"agent_id": self.agent_id})
    
    def clear_world(self):
        """Clear all world elements"""
        self.foods.clear()
        self.obstacles.clear()
        self.danger_zones.clear()
        self.forbidden_zones.clear()
    
    def generate_obstacles(self):
        """Generate static and dynamic obstacles"""
        if self.config.obstacle_count <= 0:
            return
            
        target_obstacles = self.config.obstacle_count
        attempts = 0
        max_attempts = target_obstacles * 10
        
        while len(self.obstacles) < target_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Random position
            x = random.randint(0, self.config.grid_width - 1)
            y = random.randint(0, self.config.grid_height - 1)
            pos = Position(x, y)
            
            # Check if position is valid
            if not self.is_valid_obstacle_position(pos):
                continue
            
            # Determine obstacle type
            if random.random() < self.config.moving_obstacle_chance:
                obstacle_type = ObstacleType.MOVING
                movement_pattern = random.choice(["horizontal", "vertical", "circular"])
            else:
                obstacle_type = ObstacleType.STATIC
                movement_pattern = None
            
            # Create obstacle
            obstacle = Obstacle(
                position=pos,
                obstacle_type=obstacle_type,
                spawn_time=time.time(),
                movement_pattern=movement_pattern
            )
            
            self.obstacles[pos] = obstacle
        
        print(f"🚧 Generated {len(self.obstacles)} obstacles")
    
    def generate_danger_zones(self):
        """Generate danger zones that affect gameplay"""
        zone_count = random.randint(1, 3)
        
        for _ in range(zone_count):
            # Random position avoiding spawn areas
            while True:
                x = random.randint(2, self.config.grid_width - 3)
                y = random.randint(2, self.config.grid_height - 3)
                pos = Position(x, y)
                
                # Check distance from spawn points
                safe = True
                for spawn_pos in self.snake_spawn_points.values():
                    if pos.distance_to(spawn_pos) < 5:
                        safe = False
                        break
                
                if safe:
                    radius = random.randint(2, 4)
                    self.danger_zones.append((pos, radius))
                    break
    
    def update(self):
        """Main environment update loop"""
        current_time = time.time()
        
        # Update moving obstacles
        if self.config.dynamic_obstacles:
            self.update_moving_obstacles(current_time)
        
        # Spawn food periodically
        if current_time - self.last_food_spawn > (1.0 / self.config.food_spawn_rate):
            if len(self.foods) < self.config.max_food_count:
                self.spawn_food()
                self.last_food_spawn = current_time
        
        # Remove expired foods
        self.remove_expired_foods(current_time)
        
        # Broadcast environment updates periodically
        if current_time - self.last_environment_broadcast > 0.5:  # Every 500ms
            self.broadcast_environment_update()
            self.last_environment_broadcast = current_time
        
        # Update state
        self.state = self.get_public_state()
    
    def spawn_food(self) -> bool:
        """Spawn a new food item"""
        # Find valid position
        position = self.find_valid_food_position()
        if not position:
            return False
        
        # Determine food type
        food_type = self.determine_food_type()
        
        # Calculate points value
        points_value = self.calculate_food_points(food_type)
        
        # Create food
        food = Food(
            position=position,
            food_type=food_type,
            points_value=points_value,
            spawn_time=time.time(),
            duration=30.0 + random.uniform(-5, 10)  # 25-40 seconds
        )
        
        # Add special properties for special food types
        if food_type == FoodType.SPEED_BOOST:
            food.special_data = {"speed_multiplier": 1.5, "duration": 5.0}
        elif food_type == FoodType.GROWTH_BOOST:
            food.special_data = {"growth_amount": 3}
        elif food_type == FoodType.POINTS_MULTIPLIER:
            food.special_data = {"multiplier": 2.0, "duration": 10.0}
        elif food_type == FoodType.SHRINK:
            food.special_data = {"shrink_amount": 2}
        
        self.foods[position] = food
        self.total_foods_spawned += 1
        
        print(f"🍎 Spawned {food_type.value} food at {position.x}, {position.y} (+{points_value} pts)")
        
        # Broadcast food spawn
        self.broadcast_message(MessageType.ENVIRONMENT_UPDATE, {
            "type": "food_spawned",
            "position": position.to_tuple(),
            "food_type": food_type.value,
            "points_value": points_value
        })
        
        return True
    
    def find_valid_food_position(self) -> Optional[Position]:
        """Find a valid position for food spawning"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Choose spawn zone
            if self.food_spawn_zones:
                zone_center, radius = random.choice(self.food_spawn_zones)
                # Random position within zone
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius)
                x = int(zone_center.x + distance * math.cos(angle))
                y = int(zone_center.y + distance * math.sin(angle))
            else:
                # Random position anywhere
                x = random.randint(0, self.config.grid_width - 1)
                y = random.randint(0, self.config.grid_height - 1)
            
            position = Position(x, y)
            
            if self.is_valid_food_position(position):
                return position
        
        return None
    
    def determine_food_type(self) -> FoodType:
        """Determine what type of food to spawn"""
        if random.random() < self.config.special_food_chance:
            # Special food
            special_types = [
                FoodType.SPEED_BOOST,
                FoodType.GROWTH_BOOST, 
                FoodType.POINTS_MULTIPLIER,
                FoodType.SHRINK
            ]
            return random.choice(special_types)
        else:
            return FoodType.NORMAL
    
    def calculate_food_points(self, food_type: FoodType) -> int:
        """Calculate points value for food type"""
        base_points = 10
        
        if food_type == FoodType.NORMAL:
            return base_points
        elif food_type == FoodType.SPEED_BOOST:
            return base_points + 5
        elif food_type == FoodType.GROWTH_BOOST:
            return base_points + 15
        elif food_type == FoodType.POINTS_MULTIPLIER:
            return base_points + 20
        elif food_type == FoodType.SHRINK:
            return base_points - 5  # Less points for negative effect
        
        return base_points
    
    def remove_expired_foods(self, current_time: float):
        """Remove expired food items"""
        expired_foods = []
        
        for position, food in self.foods.items():
            if food.is_expired(current_time):
                expired_foods.append(position)
        
        for position in expired_foods:
            del self.foods[position]
            self.total_foods_expired += 1
            print(f"⏰ Food expired at {position.x}, {position.y}")
    
    def update_moving_obstacles(self, current_time: float):
        """Update positions of moving obstacles"""
        if current_time - self.last_obstacle_update < 1.0:  # Update every second
            return
        
        self.last_obstacle_update = current_time
        updated_obstacles = {}
        
        for position, obstacle in self.obstacles.items():
            if obstacle.obstacle_type == ObstacleType.MOVING:
                new_position = self.calculate_obstacle_movement(obstacle, current_time)
                if new_position and new_position != position:
                    obstacle.position = new_position
                    updated_obstacles[new_position] = obstacle
                else:
                    updated_obstacles[position] = obstacle
            else:
                updated_obstacles[position] = obstacle
        
        self.obstacles = updated_obstacles
    
    def calculate_obstacle_movement(self, obstacle: Obstacle, current_time: float) -> Optional[Position]:
        """Calculate new position for moving obstacle"""
        if not obstacle.movement_pattern:
            return obstacle.position
        
        elapsed = current_time - obstacle.spawn_time
        
        try:
            if obstacle.movement_pattern == "horizontal":
                # Move back and forth horizontally
                direction = 1 if int(elapsed / 2) % 2 == 0 else -1
                new_x = obstacle.position.x + direction
                new_x = max(0, min(self.config.grid_width - 1, new_x))
                new_position = Position(new_x, obstacle.position.y)
                
            elif obstacle.movement_pattern == "vertical":
                # Move back and forth vertically
                direction = 1 if int(elapsed / 2) % 2 == 0 else -1
                new_y = obstacle.position.y + direction
                new_y = max(0, min(self.config.grid_height - 1, new_y))
                new_position = Position(obstacle.position.x, new_y)
                
            elif obstacle.movement_pattern == "circular":
                # Move in a circle
                radius = 2
                center_x = obstacle.special_data.get("center_x", obstacle.position.x)
                center_y = obstacle.special_data.get("center_y", obstacle.position.y)
                
                angle = elapsed * 0.5  # Slow rotation
                new_x = int(center_x + radius * math.cos(angle))
                new_y = int(center_y + radius * math.sin(angle))
                new_position = Position(new_x, new_y)
                
            else:
                return obstacle.position
            
            # Validate new position
            if self.is_valid_position(new_position):
                return new_position
            
        except Exception as e:
            print(f"⚠️  Obstacle movement calculation error: {e}")
        
        return obstacle.position
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= position.x < self.config.grid_width and 
                0 <= position.y < self.config.grid_height)
    
    def is_valid_food_position(self, position: Position) -> bool:
        """Check if position is valid for food placement"""
        if not self.is_valid_position(position):
            return False
        
        # Check if position is occupied
        if position in self.foods or position in self.obstacles:
            return False
        
        # Check if position conflicts with snakes
        for snake_segments in self.snakes.values():
            if position.to_tuple() in snake_segments:
                return False
        
        # Check minimum distance from other foods
        for other_position in self.foods.keys():
            if position.distance_to(other_position) < self.config.food_min_distance:
                return False
        
        return True
    
    def is_valid_obstacle_position(self, position: Position) -> bool:
        """Check if position is valid for obstacle placement"""
        if not self.is_valid_position(position):
            return False
        
        # Check forbidden zones (spawn areas)
        if position in self.forbidden_zones:
            return False
        
        # Check if position is already occupied
        if position in self.obstacles:
            return False
        
        return True
    
    def get_food_at_position(self, position: Position) -> Optional[Food]:
        """Get food at specific position"""
        if isinstance(position, tuple):
            position = Position(position[0], position[1])
        return self.foods.get(position)
    
    def remove_food_at_position(self, position: Position) -> Optional[Food]:
        """Remove and return food at position"""
        if isinstance(position, tuple):
            position = Position(position[0], position[1])
        return self.foods.pop(position, None)
    
    def get_obstacles_in_area(self, center: Position, radius: int) -> List[Obstacle]:
        """Get obstacles within radius of center"""
        obstacles_in_area = []
        
        for obstacle in self.obstacles.values():
            if center.distance_to(obstacle.position) <= radius:
                obstacles_in_area.append(obstacle)
        
        return obstacles_in_area
    
    def process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.GAME_STATE:
                self.handle_game_state_update(message.data)
                
            elif message.type == MessageType.FOOD_EATEN:
                self.handle_food_eaten(message.data)
                
            elif message.type == MessageType.COLLISION_EVENT:
                self.handle_collision_event(message.data)
                
            elif message.type == MessageType.RESET_GAME:
                self.handle_game_reset(message.data)
                
        except Exception as e:
            print(f"❌ Environment error processing message: {e}")
    
    def handle_game_state_update(self, data: Dict[str, Any]):
        """Handle game state updates"""
        update_type = data.get("type")
        
        if update_type == "config":
            config = data.get("config", {})
            self.config.grid_width = config.get("grid_width", self.config.grid_width)
            self.config.grid_height = config.get("grid_height", self.config.grid_height)
            self.config.max_food_count = config.get("max_food_count", self.config.max_food_count)
            
            print(f"📝 Environment updated config: {self.config.grid_width}x{self.config.grid_height}")
            
        elif update_type == "snake_update":
            snake_id = data.get("snake_id")
            segments = data.get("segments", [])
            
            if snake_id:
                self.snakes[snake_id] = [Position(x, y) for x, y in segments]
    
    def handle_food_eaten(self, data: Dict[str, Any]):
        """Handle food eaten events"""
        position_tuple = data.get("position")
        if position_tuple:
            position = Position(position_tuple[0], position_tuple[1])
            food = self.remove_food_at_position(position)
            
            if food:
                self.total_foods_eaten += 1
                print(f"🍽️ Food eaten at {position.x}, {position.y}")
                
                # Spawn replacement food
                self.spawn_food()
    
    def handle_collision_event(self, data: Dict[str, Any]):
        """Handle collision events"""
        collision_type = data.get("collision_type")
        position_tuple = data.get("position")
        
        if collision_type == "obstacle" and position_tuple:
            position = Position(position_tuple[0], position_tuple[1])
            obstacle = self.obstacles.get(position)
            
            if obstacle and obstacle.obstacle_type == ObstacleType.DESTRUCTIBLE:
                # Remove destructible obstacle
                del self.obstacles[position]
                print(f"💥 Destructible obstacle removed at {position.x}, {position.y}")
    
    def handle_game_reset(self, data: Dict[str, Any]):
        """Handle game reset"""
        # Update configuration if provided
        config = data.get("config", {})
        if config:
            self.config.grid_width = config.get("grid_width", self.config.grid_width)
            self.config.grid_height = config.get("grid_height", self.config.grid_height)
            self.config.max_food_count = config.get("max_food_count", self.config.max_food_count)
        
        # Regenerate world
        self.generate_world()
        
        print("🔄 Environment reset")
    
    def broadcast_environment_update(self):
        """Broadcast current environment state to all agents"""
        environment_data = {
            "foods": [food.position.to_tuple() for food in self.foods.values()],
            "obstacles": [obs.position.to_tuple() for obs in self.obstacles.values()],
            "snakes": {snake_id: [pos.to_tuple() for pos in segments] 
                      for snake_id, segments in self.snakes.items()},
            "danger_zones": [(pos.to_tuple(), radius) for pos, radius in self.danger_zones],
            "grid_size": (self.config.grid_width, self.config.grid_height)
        }
        
        self.broadcast_message(MessageType.ENVIRONMENT_UPDATE, environment_data)
    
    def get_public_state(self) -> Dict[str, Any]:
        """Get public state for other systems"""
        return {
            "grid_size": (self.config.grid_width, self.config.grid_height),
            "food_count": len(self.foods),
            "obstacle_count": len(self.obstacles),
            "total_foods_spawned": self.total_foods_spawned,
            "total_foods_eaten": self.total_foods_eaten,
            "total_foods_expired": self.total_foods_expired,
            "food_efficiency": (self.total_foods_eaten / max(1, self.total_foods_spawned)) * 100,
            "danger_zones_count": len(self.danger_zones),
            "config": {
                "max_food_count": self.config.max_food_count,
                "food_spawn_rate": self.config.food_spawn_rate,
                "obstacle_count": self.config.obstacle_count,
                "wrap_walls": self.config.wrap_walls,
                "dynamic_obstacles": self.config.dynamic_obstacles
            }
        }
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get detailed environment statistics"""
        food_types_count = {}
        for food in self.foods.values():
            food_type = food.food_type.value
            food_types_count[food_type] = food_types_count.get(food_type, 0) + 1
        
        return {
            "foods": {
                "total": len(self.foods),
                "by_type": food_types_count,
                "avg_age": sum(food.get_age(time.time()) for food in self.foods.values()) / max(1, len(self.foods))
            },
            "obstacles": {
                "total": len(self.obstacles),
                "static": sum(1 for obs in self.obstacles.values() if obs.obstacle_type == ObstacleType.STATIC),
                "moving": sum(1 for obs in self.obstacles.values() if obs.obstacle_type == ObstacleType.MOVING)
            },
            "spawn_stats": {
                "foods_spawned": self.total_foods_spawned,
                "foods_eaten": self.total_foods_eaten,
                "foods_expired": self.total_foods_expired,
                "efficiency": self.total_foods_eaten / max(1, self.total_foods_spawned)
            }
        }

if __name__ == "__main__":
    # Test the environment agent
    from multi_agent_framework import AgentRegistry
    
    config = EnvironmentConfig(
        grid_width=15,
        grid_height=15,
        max_food_count=4,
        obstacle_count=3,
        food_spawn_rate=1.0,
        dynamic_obstacles=True
    )
    
    registry = AgentRegistry()
    env_agent = EnvironmentAgent(config)
    
    registry.register_agent(env_agent)
    registry.start_all_agents()
    
    try:
        time.sleep(3)
        
        # Test food consumption
        print("\n🍎 Testing food consumption...")
        if env_agent.foods:
            food_pos = list(env_agent.foods.keys())[0]
            env_agent.send_message(MessageType.FOOD_EATEN, "environment", {
                "position": food_pos.to_tuple(),
                "player_id": "A"
            })
        
        time.sleep(5)
        
        # Check stats
        print(f"\n📊 Environment Stats:")
        stats = env_agent.get_environment_stats()
        for category, data in stats.items():
            print(f"  {category}: {data}")
            
    finally:
        registry.stop_all_agents()