"""
Visualization Agent - Rendering system for the game display
Handles all visual rendering including snakes, food, UI, effects, and animations
"""

import pygame
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from multi_agent_framework import BaseAgent, MessageType, Message

# Initialize Pygame
pygame.init()

class RenderLayer(Enum):
    BACKGROUND = 0
    GRID = 1
    ENVIRONMENT = 2
    SNAKES = 3
    FOOD = 4
    EFFECTS = 5
    UI = 6
    DEBUG = 7

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)
    
    def to_tuple_alpha(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

@dataclass 
class VisualEffect:
    effect_type: str
    position: Tuple[int, int]
    start_time: float
    duration: float
    color: Color
    size: float = 1.0
    velocity: Tuple[float, float] = (0, 0)
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RenderSettings:
    """Visual rendering configuration"""
    # Window settings
    window_width: int = 800
    window_height: int = 600
    title: str = "Snake Multi-Agent Game"
    
    # Grid settings
    cell_size: int = 20
    grid_color: Color = field(default_factory=lambda: Color(40, 40, 40))
    background_color: Color = field(default_factory=lambda: Color(10, 10, 15))
    
    # Snake colors
    snake_colors: Dict[str, Color] = field(default_factory=lambda: {
        "A": Color(0, 255, 100),    # Green
        "B": Color(255, 100, 100),  # Red
    })
    
    # Food settings
    food_color: Color = field(default_factory=lambda: Color(255, 255, 0))  # Yellow
    food_size_ratio: float = 0.8
    
    # UI settings
    ui_font_size: int = 16
    ui_color: Color = field(default_factory=lambda: Color(255, 255, 255))
    ui_margin: int = 10
    
    # Effects
    show_trail: bool = True
    show_grid: bool = True
    show_effects: bool = True
    animation_speed: float = 1.0

class VisualizationAgent(BaseAgent):
    """
    Handles all visual rendering and display for the Snake game
    """
    
    def __init__(self, settings: Optional[RenderSettings] = None):
        super().__init__("visualization")
        self.settings = settings or RenderSettings()
        
        # Pygame components
        self.screen: Optional[pygame.Surface] = None
        self.clock = pygame.time.Clock()
        self.font: Optional[pygame.font.Font] = None
        self.running = False
        
        # Game state
        self.grid_width = 20
        self.grid_height = 20
        self.game_phase = "initializing"
        
        # Rendering data
        self.snakes: Dict[str, List[Tuple[int, int]]] = {}
        self.foods: List[Tuple[int, int]] = []
        self.obstacles: List[Tuple[int, int]] = []
        self.scores: Dict[str, int] = {}
        
        # Visual effects
        self.effects: List[VisualEffect] = []
        self.trails: Dict[str, List[Tuple[Tuple[int, int], float]]] = {}
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        # Threading for pygame event loop
        self.render_thread: Optional[threading.Thread] = None
        self.quit_event = threading.Event()
        
    def initialize(self):
        """Initialize the visualization system"""
        print("🎨 Visualization Agent initializing...")
        
        # Subscribe to relevant messages
        self.subscribe_to_messages([
            MessageType.GAME_STATE,
            MessageType.ENVIRONMENT_UPDATE,
            MessageType.SCORE_UPDATE,
            MessageType.COLLISION_EVENT,
            MessageType.FOOD_EATEN,
            MessageType.RESET_GAME
        ])
        
        # Try to initialize Pygame display
        try:
            self.init_display()
            
            # Start render thread
            self.running = True
            self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
            self.render_thread.start()
            
            print("✅ Visualization Agent ready")
        except Exception as e:
            print(f"⚠️  Pygame initialization failed: {e}")
            print("🎨 Running in headless mode")
            self.running = True  # Continue without display

        # Announce readiness
        self.send_message(MessageType.AGENT_READY, "game_engine", {"agent_id": self.agent_id})
    
    def init_display(self):
        """Initialize Pygame display and resources"""
        # Check if we're on macOS and running in a thread
        import platform
        if platform.system() == "Darwin":
            print("⚠️  macOS detected - pygame may have threading issues")
            raise Exception("Pygame threading not supported on macOS")
        
        try:
            self.screen = pygame.display.set_mode((
                self.settings.window_width, 
                self.settings.window_height
            ))
            pygame.display.set_caption(self.settings.title)
            
            # Initialize font
            self.font = pygame.font.Font(None, self.settings.ui_font_size)
            
            print(f"🖥️  Display initialized: {self.settings.window_width}x{self.settings.window_height}")
            
        except pygame.error as e:
            print(f"❌ Failed to initialize display: {e}")
            self.running = False
            raise
    
    def update(self):
        """Main update - process messages even in headless mode"""
        # Process messages to keep system responsive
        if not self.screen:
            # Headless mode - just track state without rendering
            return
    
    def render_loop(self):
        """Main rendering loop running in separate thread"""
        while self.running and not self.quit_event.is_set():
            try:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit_event.set()
                        break
                    elif event.type == pygame.KEYDOWN:
                        self.handle_key_input(event.key)
                
                # Render frame
                self.render_frame()
                
                # Control frame rate (60 FPS)
                self.clock.tick(60)
                
                # Update FPS counter
                self.update_fps_counter()
                
            except Exception as e:
                print(f"❌ Render loop error: {e}")
                break
        
        # Cleanup
        pygame.quit()
    
    def render_frame(self):
        """Render a single frame"""
        if not self.screen:
            return
            
        # Clear screen
        self.screen.fill(self.settings.background_color.to_tuple())
        
        # Calculate grid offset to center the game area
        grid_pixel_width = self.grid_width * self.settings.cell_size
        grid_pixel_height = self.grid_height * self.settings.cell_size
        
        offset_x = (self.settings.window_width - grid_pixel_width) // 2
        offset_y = (self.settings.window_height - grid_pixel_height) // 2
        
        # Render layers in order
        self.render_grid(offset_x, offset_y)
        self.render_obstacles(offset_x, offset_y)
        self.render_trails(offset_x, offset_y)
        self.render_snakes(offset_x, offset_y)
        self.render_food(offset_x, offset_y)
        self.render_effects(offset_x, offset_y)
        self.render_ui()
        
        # Update display
        pygame.display.flip()
        self.frame_count += 1
    
    def render_grid(self, offset_x: int, offset_y: int):
        """Render grid lines"""
        if not self.settings.show_grid:
            return
            
        color = self.settings.grid_color.to_tuple()
        
        # Vertical lines
        for x in range(self.grid_width + 1):
            start_pos = (offset_x + x * self.settings.cell_size, offset_y)
            end_pos = (offset_x + x * self.settings.cell_size, 
                      offset_y + self.grid_height * self.settings.cell_size)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)
        
        # Horizontal lines
        for y in range(self.grid_height + 1):
            start_pos = (offset_x, offset_y + y * self.settings.cell_size)
            end_pos = (offset_x + self.grid_width * self.settings.cell_size,
                      offset_y + y * self.settings.cell_size)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)
    
    def render_obstacles(self, offset_x: int, offset_y: int):
        """Render obstacles"""
        for obs_x, obs_y in self.obstacles:
            rect = pygame.Rect(
                offset_x + obs_x * self.settings.cell_size,
                offset_y + obs_y * self.settings.cell_size,
                self.settings.cell_size,
                self.settings.cell_size
            )
            pygame.draw.rect(self.screen, (100, 50, 50), rect)
            pygame.draw.rect(self.screen, (150, 75, 75), rect, 2)
    
    def render_trails(self, offset_x: int, offset_y: int):
        """Render snake trails"""
        if not self.settings.show_trail:
            return
            
        current_time = time.time()
        
        for snake_id, trail in self.trails.items():
            if snake_id not in self.settings.snake_colors:
                continue
                
            base_color = self.settings.snake_colors[snake_id]
            
            # Remove old trail segments
            trail[:] = [(pos, timestamp) for pos, timestamp in trail 
                       if current_time - timestamp < 2.0]  # 2 second trail
            
            # Render trail segments with fading alpha
            for i, ((x, y), timestamp) in enumerate(trail):
                age = current_time - timestamp
                alpha = max(0, int(255 * (1 - age / 2.0)))  # Fade over 2 seconds
                
                if alpha > 0:
                    trail_color = (*base_color.to_tuple(), alpha)
                    trail_size = max(2, self.settings.cell_size // 4)
                    
                    center_x = offset_x + x * self.settings.cell_size + self.settings.cell_size // 2
                    center_y = offset_y + y * self.settings.cell_size + self.settings.cell_size // 2
                    
                    # Create surface with alpha
                    trail_surface = pygame.Surface((trail_size * 2, trail_size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(trail_surface, trail_color, (trail_size, trail_size), trail_size)
                    self.screen.blit(trail_surface, (center_x - trail_size, center_y - trail_size))
    
    def render_snakes(self, offset_x: int, offset_y: int):
        """Render all snakes"""
        for snake_id, segments in self.snakes.items():
            if not segments:
                continue
                
            color = self.settings.snake_colors.get(snake_id, Color(255, 255, 255))
            
            # Render each segment
            for i, (x, y) in enumerate(segments):
                rect = pygame.Rect(
                    offset_x + x * self.settings.cell_size + 1,
                    offset_y + y * self.settings.cell_size + 1,
                    self.settings.cell_size - 2,
                    self.settings.cell_size - 2
                )
                
                # Head is brighter, body segments are darker
                segment_color = color.to_tuple()
                if i == 0:  # Head
                    pygame.draw.rect(self.screen, segment_color, rect)
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
                    
                    # Add eyes to head
                    eye_size = max(2, self.settings.cell_size // 6)
                    eye_offset = self.settings.cell_size // 4
                    
                    eye1_pos = (rect.centerx - eye_offset, rect.centery - eye_offset)
                    eye2_pos = (rect.centerx + eye_offset, rect.centery - eye_offset)
                    
                    pygame.draw.circle(self.screen, (255, 255, 255), eye1_pos, eye_size)
                    pygame.draw.circle(self.screen, (255, 255, 255), eye2_pos, eye_size)
                    pygame.draw.circle(self.screen, (0, 0, 0), eye1_pos, eye_size - 1)
                    pygame.draw.circle(self.screen, (0, 0, 0), eye2_pos, eye_size - 1)
                else:  # Body
                    # Gradient effect - segments get darker towards tail
                    fade_factor = 1.0 - (i / len(segments)) * 0.5
                    faded_color = tuple(int(c * fade_factor) for c in segment_color)
                    pygame.draw.rect(self.screen, faded_color, rect)
                    pygame.draw.rect(self.screen, segment_color, rect, 1)
    
    def render_food(self, offset_x: int, offset_y: int):
        """Render food items"""
        current_time = time.time()
        
        for food_x, food_y in self.foods:
            # Pulsing animation
            pulse = math.sin(current_time * 4) * 0.2 + 1.0
            size = int(self.settings.cell_size * self.settings.food_size_ratio * pulse)
            
            center_x = offset_x + food_x * self.settings.cell_size + self.settings.cell_size // 2
            center_y = offset_y + food_y * self.settings.cell_size + self.settings.cell_size // 2
            
            # Outer glow
            glow_size = size + 4
            glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            glow_color = (*self.settings.food_color.to_tuple(), 100)
            pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surface, (center_x - glow_size, center_y - glow_size))
            
            # Main food
            pygame.draw.circle(self.screen, self.settings.food_color.to_tuple(), 
                             (center_x, center_y), size)
            pygame.draw.circle(self.screen, (255, 255, 200), 
                             (center_x, center_y), size, 2)
    
    def render_effects(self, offset_x: int, offset_y: int):
        """Render visual effects"""
        if not self.settings.show_effects:
            return
            
        current_time = time.time()
        active_effects = []
        
        for effect in self.effects:
            age = current_time - effect.start_time
            if age >= effect.duration:
                continue  # Effect expired
                
            active_effects.append(effect)
            progress = age / effect.duration
            
            if effect.effect_type == "explosion":
                self.render_explosion_effect(effect, progress, offset_x, offset_y)
            elif effect.effect_type == "food_pickup":
                self.render_food_pickup_effect(effect, progress, offset_x, offset_y)
            elif effect.effect_type == "death":
                self.render_death_effect(effect, progress, offset_x, offset_y)
        
        self.effects = active_effects
    
    def render_explosion_effect(self, effect: VisualEffect, progress: float, offset_x: int, offset_y: int):
        """Render explosion effect"""
        alpha = int(255 * (1 - progress))
        size = int(effect.size * (1 + progress * 3))
        
        center_x = offset_x + effect.position[0] * self.settings.cell_size + self.settings.cell_size // 2
        center_y = offset_y + effect.position[1] * self.settings.cell_size + self.settings.cell_size // 2
        
        # Create explosion surface
        explosion_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        explosion_color = (*effect.color.to_tuple(), alpha)
        
        # Multiple circles for explosion effect
        for i in range(3):
            circle_size = size - i * (size // 4)
            if circle_size > 0:
                circle_alpha = alpha // (i + 1)
                circle_color = (*effect.color.to_tuple(), circle_alpha)
                pygame.draw.circle(explosion_surface, circle_color, (size, size), circle_size)
        
        self.screen.blit(explosion_surface, (center_x - size, center_y - size))
    
    def render_food_pickup_effect(self, effect: VisualEffect, progress: float, offset_x: int, offset_y: int):
        """Render food pickup effect"""
        # Rising text effect
        alpha = int(255 * (1 - progress))
        rise_offset = int(progress * 30)  # Rise 30 pixels
        
        pos_x = offset_x + effect.position[0] * self.settings.cell_size + self.settings.cell_size // 2
        pos_y = offset_y + effect.position[1] * self.settings.cell_size - rise_offset
        
        points_text = f"+{effect.data.get('points', 10)}"
        text_surface = self.font.render(points_text, True, effect.color.to_tuple())
        
        # Add alpha
        text_surface.set_alpha(alpha)
        
        # Center text
        text_rect = text_surface.get_rect(center=(pos_x, pos_y))
        self.screen.blit(text_surface, text_rect)
    
    def render_death_effect(self, effect: VisualEffect, progress: float, offset_x: int, offset_y: int):
        """Render death effect"""
        # Expanding red circle
        alpha = int(255 * (1 - progress))
        size = int(effect.size * (1 + progress * 2))
        
        center_x = offset_x + effect.position[0] * self.settings.cell_size + self.settings.cell_size // 2
        center_y = offset_y + effect.position[1] * self.settings.cell_size + self.settings.cell_size // 2
        
        death_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        death_color = (255, 50, 50, alpha)
        
        pygame.draw.circle(death_surface, death_color, (size, size), size)
        pygame.draw.circle(death_surface, (255, 100, 100, alpha), (size, size), size, 3)
        
        self.screen.blit(death_surface, (center_x - size, center_y - size))
    
    def render_ui(self):
        """Render user interface elements"""
        margin = self.settings.ui_margin
        color = self.settings.ui_color.to_tuple()
        
        # Game phase
        phase_text = f"Phase: {self.game_phase.title()}"
        phase_surface = self.font.render(phase_text, True, color)
        self.screen.blit(phase_surface, (margin, margin))
        
        # Scores
        y_offset = margin + 25
        for snake_id, score in self.scores.items():
            snake_color = self.settings.snake_colors.get(snake_id, Color(255, 255, 255))
            score_text = f"Snake {snake_id}: {score}"
            score_surface = self.font.render(score_text, True, snake_color.to_tuple())
            self.screen.blit(score_surface, (margin, y_offset))
            y_offset += 20
        
        # FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        fps_surface = self.font.render(fps_text, True, color)
        fps_rect = fps_surface.get_rect(topright=(self.settings.window_width - margin, margin))
        self.screen.blit(fps_surface, fps_rect)
        
        # Frame count
        frame_text = f"Frame: {self.frame_count}"
        frame_surface = self.font.render(frame_text, True, color)
        frame_rect = frame_surface.get_rect(topright=(self.settings.window_width - margin, margin + 20))
        self.screen.blit(frame_surface, frame_rect)
    
    def handle_key_input(self, key):
        """Handle keyboard input for debug/control"""
        if key == pygame.K_g:
            self.settings.show_grid = not self.settings.show_grid
        elif key == pygame.K_t:
            self.settings.show_trail = not self.settings.show_trail
        elif key == pygame.K_e:
            self.settings.show_effects = not self.settings.show_effects
        elif key == pygame.K_r:
            # Send reset request
            self.send_message(MessageType.RESET_GAME, "game_engine", {"source": "visualization"})
    
    def update_fps_counter(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def add_effect(self, effect_type: str, position: Tuple[int, int], 
                   duration: float = 1.0, color: Optional[Color] = None, 
                   size: float = 20.0, **kwargs):
        """Add a visual effect"""
        if color is None:
            color = Color(255, 255, 255)
            
        effect = VisualEffect(
            effect_type=effect_type,
            position=position,
            start_time=time.time(),
            duration=duration,
            color=color,
            size=size,
            data=kwargs
        )
        
        self.effects.append(effect)
    
    def add_trail_point(self, snake_id: str, position: Tuple[int, int]):
        """Add a point to snake trail"""
        if snake_id not in self.trails:
            self.trails[snake_id] = []
            
        self.trails[snake_id].append((position, time.time()))
        
        # Limit trail length
        max_trail_length = 20
        if len(self.trails[snake_id]) > max_trail_length:
            self.trails[snake_id].pop(0)
    
    def process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.GAME_STATE:
                self.handle_game_state_update(message.data)
                
            elif message.type == MessageType.ENVIRONMENT_UPDATE:
                self.handle_environment_update(message.data)
                
            elif message.type == MessageType.SCORE_UPDATE:
                self.handle_score_update(message.data)
                
            elif message.type == MessageType.COLLISION_EVENT:
                self.handle_collision_event(message.data)
                
            elif message.type == MessageType.FOOD_EATEN:
                self.handle_food_eaten(message.data)
                
            elif message.type == MessageType.RESET_GAME:
                self.handle_game_reset(message.data)
                
        except Exception as e:
            print(f"❌ Visualization error processing message: {e}")
    
    def handle_game_state_update(self, data: Dict[str, Any]):
        """Handle game state updates"""
        update_type = data.get("type")
        
        if update_type == "config":
            config = data.get("config", {})
            self.grid_width = config.get("grid_width", self.grid_width)
            self.grid_height = config.get("grid_height", self.grid_height)
            
        elif update_type == "update":
            self.game_phase = data.get("phase", self.game_phase)
            players = data.get("players", {})
            
            # Update scores
            for player_id, player_data in players.items():
                self.scores[player_id] = player_data.get("score", 0)
                
        elif update_type == "snake_update":
            snake_id = data.get("snake_id")
            segments = data.get("segments", [])
            
            if snake_id:
                self.snakes[snake_id] = segments
                
                # Add trail point for head
                if segments and self.settings.show_trail:
                    self.add_trail_point(snake_id, segments[0])
    
    def handle_environment_update(self, data: Dict[str, Any]):
        """Handle environment updates"""
        self.foods = data.get("foods", [])
        self.obstacles = data.get("obstacles", [])
    
    def handle_score_update(self, data: Dict[str, Any]):
        """Handle score updates"""
        player_id = data.get("player_id")
        new_score = data.get("new_score", 0)
        
        if player_id:
            self.scores[player_id] = new_score
    
    def handle_collision_event(self, data: Dict[str, Any]):
        """Handle collision events"""
        player_id = data.get("player_id")
        position = data.get("position", (0, 0))
        collision_type = data.get("collision_type", "unknown")
        
        # Add death effect
        self.add_effect("death", position, duration=2.0, 
                       color=Color(255, 50, 50), size=30)
        
        print(f"💥 Visualizing collision: {player_id} at {position}")
    
    def handle_food_eaten(self, data: Dict[str, Any]):
        """Handle food eaten events"""
        position = data.get("position", (0, 0))
        points = data.get("points", 10)
        
        # Add food pickup effect
        self.add_effect("food_pickup", position, duration=1.5,
                       color=Color(255, 255, 0), size=15, points=points)
        
        # Remove food from display
        if position in self.foods:
            self.foods.remove(position)
    
    def handle_game_reset(self, data: Dict[str, Any]):
        """Handle game reset"""
        # Clear all visual state
        self.snakes.clear()
        self.foods.clear()
        self.obstacles.clear()
        self.trails.clear()
        self.effects.clear()
        self.scores.clear()
        
        self.game_phase = "ready"
        
        print("🔄 Visualization reset")
    
    def stop(self):
        """Stop the visualization agent"""
        self.running = False
        self.quit_event.set()
        
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=2.0)
        
        super().stop()
    
    def get_public_state(self) -> Dict[str, Any]:
        """Get public state"""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "effects_count": len(self.effects),
            "trails_count": sum(len(trail) for trail in self.trails.values()),
            "display_initialized": self.screen is not None,
            "grid_size": (self.grid_width, self.grid_height),
            "window_size": (self.settings.window_width, self.settings.window_height)
        }

if __name__ == "__main__":
    # Test the visualization agent
    from multi_agent_framework import AgentRegistry
    
    settings = RenderSettings(
        window_width=600,
        window_height=600,
        cell_size=25
    )
    
    registry = AgentRegistry()
    viz_agent = VisualizationAgent(settings)
    
    registry.register_agent(viz_agent)
    registry.start_all_agents()
    
    try:
        # Test with some sample data
        time.sleep(1)
        
        # Simulate snake update
        viz_agent.send_message(MessageType.GAME_STATE, "visualization", {
            "type": "snake_update",
            "snake_id": "A",
            "segments": [(5, 5), (4, 5), (3, 5)]
        })
        
        # Simulate food
        viz_agent.send_message(MessageType.ENVIRONMENT_UPDATE, "visualization", {
            "foods": [(10, 10), (15, 7)],
            "obstacles": [(8, 8)]
        })
        
        # Keep running until user closes window
        while viz_agent.running and not viz_agent.quit_event.is_set():
            time.sleep(0.1)
            
    finally:
        registry.stop_all_agents()