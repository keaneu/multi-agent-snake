"""
Multi-Agent Snake Game Framework
Based on the architecture: Game Engine, Snake Logic, AI Training, Visualization, Environment
"""

import threading
import queue
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

class MessageType(Enum):
    GAME_STATE = "game_state"
    MOVE_COMMAND = "move_command" 
    COLLISION_EVENT = "collision_event"
    FOOD_EATEN = "food_eaten"
    SCORE_UPDATE = "score_update"
    RENDER_REQUEST = "render_request"
    AI_DECISION = "ai_decision"
    ENVIRONMENT_UPDATE = "environment_update"
    GAME_OVER = "game_over"
    RESET_GAME = "reset_game"

@dataclass
class Message:
    type: MessageType
    sender: str
    recipient: str
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'type': self.type.value,
            'sender': self.sender,
            'recipient': self.recipient,
            'data': self.data,
            'timestamp': self.timestamp
        }

class MessageBus:
    """Central communication hub for all agents"""
    
    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[MessageType, List[str]] = {}
        self.running = False
        self.message_log = []
        self.max_log_size = 1000
    
    def register_agent(self, agent: 'BaseAgent'):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        self.message_queues[agent.agent_id] = queue.Queue()
        agent.set_message_bus(self)
        print(f"Registered agent: {agent.agent_id}")
    
    def subscribe(self, agent_id: str, message_type: MessageType):
        """Subscribe agent to specific message types"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        if agent_id not in self.subscribers[message_type]:
            self.subscribers[message_type].append(agent_id)
    
    def unsubscribe(self, agent_id: str, message_type: MessageType):
        """Unsubscribe agent from message type"""
        if message_type in self.subscribers:
            if agent_id in self.subscribers[message_type]:
                self.subscribers[message_type].remove(agent_id)
    
    def send_message(self, message: Message):
        """Send message to specific recipient or broadcast to subscribers"""
        self._log_message(message)
        
        if message.recipient == "broadcast":
            # Broadcast to all subscribers of this message type
            subscribers = self.subscribers.get(message.type, [])
            for agent_id in subscribers:
                if agent_id != message.sender:  # Don't send to sender
                    self.message_queues[agent_id].put(message)
        else:
            # Send to specific recipient
            if message.recipient in self.message_queues:
                self.message_queues[message.recipient].put(message)
            else:
                print(f"Warning: Recipient {message.recipient} not found")
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all pending messages for an agent"""
        messages = []
        if agent_id in self.message_queues:
            try:
                while True:
                    message = self.message_queues[agent_id].get_nowait()
                    messages.append(message)
            except queue.Empty:
                pass
        return messages
    
    def _log_message(self, message: Message):
        """Log message for debugging"""
        self.message_log.append(message.to_dict())
        if len(self.message_log) > self.max_log_size:
            self.message_log.pop(0)
    
    def get_message_log(self, last_n: int = 10) -> List[Dict]:
        """Get recent messages for debugging"""
        return self.message_log[-last_n:]

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_bus: Optional[MessageBus] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.subscriptions: List[MessageType] = []
        self.state = {}
        
    def set_message_bus(self, message_bus: MessageBus):
        """Set the message bus for communication"""
        self.message_bus = message_bus
        
    def subscribe_to_messages(self, message_types: List[MessageType]):
        """Subscribe to specific message types"""
        self.subscriptions = message_types
        for msg_type in message_types:
            if self.message_bus:
                self.message_bus.subscribe(self.agent_id, msg_type)
    
    def send_message(self, msg_type: MessageType, recipient: str, data: Dict[str, Any]):
        """Send a message via the message bus"""
        if self.message_bus:
            message = Message(msg_type, self.agent_id, recipient, data)
            self.message_bus.send_message(message)
    
    def broadcast_message(self, msg_type: MessageType, data: Dict[str, Any]):
        """Broadcast a message to all subscribers"""
        self.send_message(msg_type, "broadcast", data)
    
    def get_messages(self) -> List[Message]:
        """Get pending messages from the message bus"""
        if self.message_bus:
            return self.message_bus.get_messages(self.agent_id)
        return []
    
    def start(self):
        """Start the agent in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print(f"Started agent: {self.agent_id}")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print(f"Stopped agent: {self.agent_id}")
    
    def _run_loop(self):
        """Main execution loop for the agent"""
        self.initialize()
        
        while self.running:
            try:
                # Process incoming messages
                messages = self.get_messages()
                for message in messages:
                    self.process_message(message)
                
                # Execute agent logic
                self.update()
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                print(f"Error in agent {self.agent_id}: {e}")
                import traceback
                traceback.print_exc()
    
    @abstractmethod
    def initialize(self):
        """Initialize the agent - called once when started"""
        pass
    
    @abstractmethod
    def update(self):
        """Main update logic - called continuously while running"""
        pass
    
    @abstractmethod
    def process_message(self, message: Message):
        """Process incoming messages"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.copy()

class AgentRegistry:
    """Registry to manage all agents in the system"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = MessageBus()
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
    
    def start_all_agents(self):
        """Start all registered agents"""
        for agent in self.agents.values():
            agent.start()
        print(f"Started {len(self.agents)} agents")
    
    def stop_all_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            agent.stop()
        print("Stopped all agents")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get state of all agents"""
        return {
            agent_id: agent.get_state() 
            for agent_id, agent in self.agents.items()
        }
    
    def send_system_message(self, msg_type: MessageType, data: Dict[str, Any]):
        """Send message from system to all agents"""
        message = Message(msg_type, "system", "broadcast", data)
        self.message_bus.send_message(message)

# Example implementations of specific agent types

class GameEngineAgent(BaseAgent):
    """Coordinates the overall game state and flow"""
    
    def __init__(self):
        super().__init__("game_engine")
        self.game_state = {
            "running": False,
            "paused": False,
            "score": {"A": 0, "B": 0},
            "round": 0,
            "fps": 60
        }
        
    def initialize(self):
        self.subscribe_to_messages([
            MessageType.COLLISION_EVENT,
            MessageType.FOOD_EATEN,
            MessageType.GAME_OVER,
            MessageType.RESET_GAME
        ])
        self.state = self.game_state
        
    def update(self):
        if self.game_state["running"] and not self.game_state["paused"]:
            # Broadcast current game state periodically
            self.broadcast_message(MessageType.GAME_STATE, self.game_state)
            time.sleep(1.0 / self.game_state["fps"])  # Maintain FPS
    
    def process_message(self, message: Message):
        if message.type == MessageType.FOOD_EATEN:
            snake_id = message.data.get("snake_id", "A")
            self.game_state["score"][snake_id] += message.data.get("points", 10)
            self.broadcast_message(MessageType.SCORE_UPDATE, self.game_state["score"])
            
        elif message.type == MessageType.COLLISION_EVENT:
            self.handle_collision(message.data)
            
        elif message.type == MessageType.RESET_GAME:
            self.reset_game()
    
    def handle_collision(self, data):
        """Handle collision events"""
        self.game_state["running"] = False
        self.broadcast_message(MessageType.GAME_OVER, {"reason": "collision", "details": data})
    
    def reset_game(self):
        """Reset game to initial state"""
        self.game_state["score"] = {"A": 0, "B": 0}
        self.game_state["round"] += 1
        self.game_state["running"] = True
        self.game_state["paused"] = False

class SnakeLogicAgent(BaseAgent):
    """Handles snake movement, collision detection, and growth"""
    
    def __init__(self, snake_id: str = "A"):
        super().__init__(f"snake_logic_{snake_id}")
        self.snake_id = snake_id
        
    def initialize(self):
        self.subscribe_to_messages([
            MessageType.MOVE_COMMAND,
            MessageType.GAME_STATE,
            MessageType.RESET_GAME
        ])
        
    def update(self):
        # Snake logic updates happen in response to messages
        pass
    
    def process_message(self, message: Message):
        if message.type == MessageType.MOVE_COMMAND:
            if message.data.get("snake_id") == self.snake_id:
                self.move_snake(message.data.get("direction"))

    def move_snake(self, direction):
        """Move snake and check for collisions"""
        # Implementation would go here
        pass

if __name__ == "__main__":
    # Example usage
    registry = AgentRegistry()
    
    # Create agents
    game_engine = GameEngineAgent()
    snake_agent_a = SnakeLogicAgent("A")
    snake_agent_b = SnakeLogicAgent("B")
    
    # Register agents
    registry.register_agent(game_engine)
    registry.register_agent(snake_agent_a)
    registry.register_agent(snake_agent_b)
    
    # Start system
    registry.start_all_agents()
    
    try:
        # Run for a bit
        time.sleep(5)
        
        # Send test message
        registry.send_system_message(
            MessageType.RESET_GAME, 
            {"reason": "manual_reset"}
        )
        
        time.sleep(2)
        
    finally:
        # Clean shutdown
        registry.stop_all_agents()