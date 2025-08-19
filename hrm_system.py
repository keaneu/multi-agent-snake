"""
Hierarchical Reasoning Model (HRM) System for Snake AI
Implements multi-level goal hierarchies, temporal options, and goal-conditional policies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

class GoalType(Enum):
    """Types of goals in the hierarchy"""
    META = "meta"           # Long-term objectives (maximize score)
    STRATEGIC = "strategic" # Medium-term plans (survive, acquire food)
    TACTICAL = "tactical"   # Short-term actions (avoid wall, move toward food)

class OptionStatus(Enum):
    """Status of temporal options"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Goal:
    """Represents a goal in the hierarchy"""
    goal_id: str
    goal_type: GoalType
    description: str
    priority: float
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    time_horizon: int = 1  # Steps to completion
    reward_weight: float = 1.0
    is_active: bool = False
    progress: float = 0.0

@dataclass
class TemporalOption:
    """Represents a temporal option (macro-action)"""
    option_id: str
    name: str
    goal_id: str
    duration: int  # Expected duration in steps
    min_duration: int = 1
    max_duration: int = 50
    status: OptionStatus = OptionStatus.INACTIVE
    start_time: int = 0
    completion_condition: str = ""  # Function name or condition
    primitive_actions: List[int] = field(default_factory=list)
    success_rate: float = 0.0
    execution_count: int = 0

class GoalHierarchy:
    """Manages the hierarchical goal structure"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.goal_relationships: Dict[str, List[str]] = {}
        self.active_goals: List[str] = []
        self.goal_stack: deque = deque()  # Stack for goal execution
        
        self._initialize_default_goals()
    
    def _initialize_default_goals(self):
        """Initialize the default goal hierarchy for Snake game"""
        
        # META LEVEL GOALS
        meta_goal = Goal(
            goal_id="maximize_score",
            goal_type=GoalType.META,
            description="Maximize total game score",
            priority=1.0,
            completion_criteria={"score_threshold": 50},
            time_horizon=1000,
            reward_weight=1.0
        )
        self.add_goal(meta_goal)
        
        # STRATEGIC LEVEL GOALS
        strategic_goals = [
            Goal(
                goal_id="long_term_survival",
                goal_type=GoalType.STRATEGIC,
                description="Survive for extended periods",
                priority=0.9,
                parent_goal="maximize_score",
                completion_criteria={"survival_time": 100},
                time_horizon=100,
                reward_weight=0.8
            ),
            Goal(
                goal_id="efficient_food_collection",
                goal_type=GoalType.STRATEGIC,
                description="Collect food efficiently",
                priority=0.8,
                parent_goal="maximize_score",
                completion_criteria={"food_collected": 5},
                time_horizon=50,
                reward_weight=1.2
            ),
            Goal(
                goal_id="territory_control",
                goal_type=GoalType.STRATEGIC,
                description="Control optimal areas of the game space",
                priority=0.7,
                parent_goal="maximize_score",
                completion_criteria={"area_coverage": 0.3},
                time_horizon=75,
                reward_weight=0.6
            )
        ]
        
        for goal in strategic_goals:
            self.add_goal(goal)
        
        # TACTICAL LEVEL GOALS
        tactical_goals = [
            Goal(
                goal_id="avoid_immediate_collision",
                goal_type=GoalType.TACTICAL,
                description="Avoid immediate collision with walls/obstacles/self",
                priority=1.0,
                parent_goal="long_term_survival",
                completion_criteria={"collision_risk": 0.1},
                time_horizon=3,
                reward_weight=2.0
            ),
            Goal(
                goal_id="move_toward_nearest_food",
                goal_type=GoalType.TACTICAL,
                description="Move towards the nearest accessible food",
                priority=0.8,
                parent_goal="efficient_food_collection",
                completion_criteria={"distance_to_food": "decreasing"},
                time_horizon=10,
                reward_weight=1.0
            ),
            Goal(
                goal_id="maintain_safe_distance",
                goal_type=GoalType.TACTICAL,
                description="Keep safe distance from walls and obstacles",
                priority=0.6,
                parent_goal="long_term_survival",
                completion_criteria={"min_distance": 2},
                time_horizon=5,
                reward_weight=0.5
            ),
            Goal(
                goal_id="optimize_path_efficiency",
                goal_type=GoalType.TACTICAL,
                description="Take efficient paths to objectives",
                priority=0.7,
                parent_goal="efficient_food_collection",
                completion_criteria={"path_efficiency": 0.8},
                time_horizon=15,
                reward_weight=0.8
            ),
            Goal(
                goal_id="explore_unknown_areas",
                goal_type=GoalType.TACTICAL,
                description="Explore areas with potential food spawns",
                priority=0.5,
                parent_goal="territory_control",
                completion_criteria={"exploration_progress": 0.1},
                time_horizon=20,
                reward_weight=0.4
            )
        ]
        
        for goal in tactical_goals:
            self.add_goal(goal)
    
    def add_goal(self, goal: Goal):
        """Add a goal to the hierarchy"""
        self.goals[goal.goal_id] = goal
        
        # Update relationships
        if goal.parent_goal:
            if goal.parent_goal not in self.goal_relationships:
                self.goal_relationships[goal.parent_goal] = []
            self.goal_relationships[goal.parent_goal].append(goal.goal_id)
            
            # Add to parent's subgoals
            if goal.parent_goal in self.goals:
                self.goals[goal.parent_goal].subgoals.append(goal.goal_id)
    
    def activate_goal(self, goal_id: str) -> bool:
        """Activate a goal and its necessary dependencies"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        goal.is_active = True
        
        if goal_id not in self.active_goals:
            self.active_goals.append(goal_id)
            self.goal_stack.append(goal_id)
        
        # Activate parent goals if necessary
        if goal.parent_goal and not self.goals[goal.parent_goal].is_active:
            self.activate_goal(goal.parent_goal)
        
        return True
    
    def complete_goal(self, goal_id: str, success: bool = True) -> float:
        """Mark a goal as completed and return reward"""
        if goal_id not in self.goals:
            return 0.0
        
        goal = self.goals[goal_id]
        goal.is_active = False
        goal.progress = 1.0 if success else 0.0
        
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        
        # Remove from stack
        if goal_id in self.goal_stack:
            self.goal_stack.remove(goal_id)
        
        # Calculate reward
        reward = goal.reward_weight * (1.0 if success else -0.5)
        
        # Check if parent goals can be progressed
        if goal.parent_goal and success:
            parent = self.goals[goal.parent_goal]
            completed_subgoals = sum(1 for sg in parent.subgoals 
                                   if self.goals[sg].progress >= 1.0)
            parent.progress = completed_subgoals / len(parent.subgoals)
        
        return reward
    
    def get_active_goals_by_priority(self) -> List[Goal]:
        """Get active goals sorted by priority"""
        active = [self.goals[gid] for gid in self.active_goals]
        return sorted(active, key=lambda g: g.priority, reverse=True)
    
    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress of a specific goal"""
        if goal_id in self.goals:
            self.goals[goal_id].progress = min(1.0, max(0.0, progress))

class TemporalOptions:
    """Manages temporal options (macro-actions) for hierarchical planning"""
    
    def __init__(self):
        self.options: Dict[str, TemporalOption] = {}
        self.active_option: Optional[TemporalOption] = None
        self.option_history: deque = deque(maxlen=100)
        self.current_step = 0
        
        self._initialize_default_options()
    
    def _initialize_default_options(self):
        """Initialize default temporal options for Snake game"""
        
        options = [
            # SURVIVAL OPTIONS
            TemporalOption(
                option_id="emergency_escape",
                name="Emergency Escape Sequence",
                goal_id="avoid_immediate_collision",
                duration=3,
                min_duration=1,
                max_duration=5,
                completion_condition="collision_risk_reduced"
            ),
            TemporalOption(
                option_id="safe_navigation",
                name="Navigate Safely",
                goal_id="maintain_safe_distance",
                duration=8,
                min_duration=3,
                max_duration=15,
                completion_condition="safe_distance_maintained"
            ),
            
            # FOOD ACQUISITION OPTIONS
            TemporalOption(
                option_id="direct_food_approach",
                name="Direct Approach to Food",
                goal_id="move_toward_nearest_food",
                duration=10,
                min_duration=3,
                max_duration=20,
                completion_condition="food_acquired_or_unreachable"
            ),
            TemporalOption(
                option_id="strategic_food_hunt",
                name="Strategic Food Hunting",
                goal_id="efficient_food_collection",
                duration=25,
                min_duration=10,
                max_duration=40,
                completion_condition="multiple_food_acquired"
            ),
            
            # EXPLORATION OPTIONS
            TemporalOption(
                option_id="area_exploration",
                name="Systematic Area Exploration",
                goal_id="explore_unknown_areas",
                duration=20,
                min_duration=8,
                max_duration=35,
                completion_condition="area_explored"
            ),
            TemporalOption(
                option_id="perimeter_patrol",
                name="Patrol Game Perimeter",
                goal_id="territory_control",
                duration=30,
                min_duration=15,
                max_duration=50,
                completion_condition="perimeter_covered"
            )
        ]
        
        for option in options:
            self.options[option.option_id] = option
    
    def select_option_for_goal(self, goal_id: str) -> Optional[TemporalOption]:
        """Select the best temporal option for a given goal"""
        candidates = [opt for opt in self.options.values() 
                     if opt.goal_id == goal_id and opt.status == OptionStatus.INACTIVE]
        
        if not candidates:
            return None
        
        # Select based on success rate and execution count (exploration vs exploitation)
        best_option = max(candidates, 
                         key=lambda o: o.success_rate + 0.1 / (o.execution_count + 1))
        
        return best_option
    
    def initiate_option(self, option: TemporalOption) -> bool:
        """Initiate execution of a temporal option"""
        if self.active_option:
            self.terminate_option(success=False)
        
        option.status = OptionStatus.ACTIVE
        option.start_time = self.current_step
        option.execution_count += 1
        self.active_option = option
        
        return True
    
    def continue_option(self) -> bool:
        """Continue executing the current active option"""
        if not self.active_option:
            return False
        
        option = self.active_option
        elapsed = self.current_step - option.start_time
        
        # Check if option should terminate
        if elapsed >= option.max_duration:
            return self.terminate_option(success=False)
        
        # Check completion condition (simplified)
        if self._check_completion_condition(option):
            return self.terminate_option(success=True)
        
        return True
    
    def terminate_option(self, success: bool) -> bool:
        """Terminate the current active option"""
        if not self.active_option:
            return False
        
        option = self.active_option
        option.status = OptionStatus.COMPLETED if success else OptionStatus.FAILED
        
        # Update success rate
        total_attempts = option.execution_count
        if success:
            option.success_rate = ((option.success_rate * (total_attempts - 1)) + 1) / total_attempts
        else:
            option.success_rate = (option.success_rate * (total_attempts - 1)) / total_attempts
        
        # Add to history
        self.option_history.append({
            'option_id': option.option_id,
            'success': success,
            'duration': self.current_step - option.start_time,
            'step': self.current_step
        })
        
        # Reset status
        option.status = OptionStatus.INACTIVE
        self.active_option = None
        
        return True
    
    def _check_completion_condition(self, option: TemporalOption) -> bool:
        """Check if an option's completion condition is met"""
        # This would be implemented with actual game state checking
        # For now, return False to let options run their course
        return False
    
    def update_step(self):
        """Update the current step counter"""
        self.current_step += 1

class GoalConditionalPolicy(nn.Module):
    """Neural network that generates actions conditioned on current goals"""
    
    def __init__(self, state_size: int = 33, goal_embedding_size: int = 16, 
                 hidden_size: int = 128, action_size: int = 3):
        super(GoalConditionalPolicy, self).__init__()
        
        self.state_size = state_size
        self.goal_embedding_size = goal_embedding_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Goal embeddings
        self.goal_types = ["maximize_score", "long_term_survival", "efficient_food_collection", 
                          "territory_control", "avoid_immediate_collision", "move_toward_nearest_food",
                          "maintain_safe_distance", "optimize_path_efficiency", "explore_unknown_areas"]
        
        self.goal_embedding = nn.Embedding(len(self.goal_types), goal_embedding_size)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Goal-conditioned attention
        self.goal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Policy network
        combined_size = hidden_size + goal_embedding_size
        self.policy_network = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value network for each goal type
        self.value_networks = nn.ModuleDict({
            goal_type: nn.Sequential(
                nn.Linear(combined_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            ) for goal_type in self.goal_types
        })
    
    def forward(self, state: torch.Tensor, active_goals: List[str], 
                goal_priorities: List[float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass conditioned on active goals
        
        Args:
            state: Current game state
            active_goals: List of active goal IDs
            goal_priorities: Priority weights for active goals
            
        Returns:
            action_logits: Policy output for actions
            goal_values: Value estimates for each active goal
        """
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Encode state
        state_features = self.state_encoder(state)
        
        # Handle goals
        if not active_goals:
            # Default goal if none active
            active_goals = ["long_term_survival"]
            goal_priorities = [1.0]
        
        # Get goal embeddings
        goal_indices = []
        valid_priorities = []
        for i, goal in enumerate(active_goals):
            if goal in self.goal_types:
                goal_indices.append(self.goal_types.index(goal))
                valid_priorities.append(goal_priorities[i] if i < len(goal_priorities) else 1.0)
        
        if not goal_indices:
            goal_indices = [1]  # Default to survival goal
            valid_priorities = [1.0]
        
        goal_idx_tensor = torch.LongTensor(goal_indices).to(state.device)
        goal_embeddings = self.goal_embedding(goal_idx_tensor)
        
        # Weight goal embeddings by priority
        priority_weights = torch.FloatTensor(valid_priorities).to(state.device)
        priority_weights = priority_weights / priority_weights.sum()  # Normalize
        
        # Weighted combination of goal embeddings
        if len(goal_embeddings.shape) == 2:
            weighted_goal_embedding = torch.sum(
                goal_embeddings * priority_weights.unsqueeze(1), dim=0, keepdim=True
            )
        else:
            weighted_goal_embedding = goal_embeddings * priority_weights
        
        # Expand to match batch size
        if weighted_goal_embedding.size(0) != batch_size:
            weighted_goal_embedding = weighted_goal_embedding.expand(batch_size, -1)
        
        # Combine state and goal features
        combined_features = torch.cat([state_features, weighted_goal_embedding], dim=1)
        
        # Generate action logits
        action_logits = self.policy_network(combined_features)
        
        # Generate value estimates for active goals
        goal_values = {}
        for goal in active_goals:
            if goal in self.goal_types and goal in self.value_networks:
                goal_values[goal] = self.value_networks[goal](combined_features)
        
        return action_logits, goal_values

class HierarchicalReasoningModel:
    """Main HRM system that coordinates goals, options, and policies"""
    
    def __init__(self, state_size: int = 33):
        self.goal_hierarchy = GoalHierarchy()
        self.temporal_options = TemporalOptions()
        self.policy = GoalConditionalPolicy(state_size=state_size)
        
        # HRM state
        self.current_state = None
        self.step_count = 0
        self.goal_completion_history = deque(maxlen=1000)
        self.performance_metrics = {
            'goals_completed': 0,
            'goals_failed': 0,
            'average_option_success_rate': 0.0,
            'hierarchical_reward': 0.0
        }
        
        # Initialize with basic survival goals
        self.goal_hierarchy.activate_goal("maximize_score")
        self.goal_hierarchy.activate_goal("long_term_survival")
        
        print("🧠 HRM System initialized with hierarchical goals and temporal options")
        print(f"   📊 Goals: {len(self.goal_hierarchy.goals)} | Options: {len(self.temporal_options.options)}")
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, Dict[str, Any]]:
        """Select action using hierarchical reasoning"""
        self.current_state = state
        self.step_count += 1
        self.temporal_options.update_step()
        
        # Update goal states based on current situation
        self._update_goal_states(state)
        
        # Get active goals and their priorities
        active_goals = self.goal_hierarchy.get_active_goals_by_priority()
        goal_ids = [g.goal_id for g in active_goals[:3]]  # Top 3 goals
        goal_priorities = [g.priority * (1 + g.progress) for g in active_goals[:3]]
        
        # Check if we need to select/continue temporal option
        self._manage_temporal_options(goal_ids)
        
        # Generate action using goal-conditional policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, goal_values = self.policy(state_tensor, goal_ids, goal_priorities)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(self.policy.action_size)
                decision_type = "exploration"
            else:
                action = int(torch.argmax(action_logits, dim=1).item())
                decision_type = "hierarchical_policy"
        
        # Create explanation
        explanation = {
            'decision_type': decision_type,
            'active_goals': goal_ids,
            'goal_priorities': goal_priorities,
            'active_option': self.temporal_options.active_option.option_id if self.temporal_options.active_option else None,
            'goal_values': {k: v.item() for k, v in goal_values.items()},
            'hierarchical_reasoning': True
        }
        
        return action, explanation
    
    def _update_goal_states(self, state: np.ndarray):
        """Update goal states based on current game situation"""
        # Extract game information from state
        snake_length = self._estimate_snake_length(state)
        collision_risk = self._estimate_collision_risk(state)
        food_distance = self._estimate_food_distance(state)
        
        # Update goal progress based on game state
        survival_goal = self.goal_hierarchy.goals.get("long_term_survival")
        if survival_goal:
            # Progress based on survival time
            survival_progress = min(1.0, self.step_count / 100.0)
            survival_goal.progress = survival_progress
        
        food_goal = self.goal_hierarchy.goals.get("move_toward_nearest_food")
        if food_goal:
            # Progress based on getting closer to food
            if hasattr(self, '_last_food_distance'):
                if food_distance < self._last_food_distance:
                    food_goal.progress = min(1.0, food_goal.progress + 0.1)
            self._last_food_distance = food_distance
        
        # Activate emergency goals if needed
        if collision_risk > 0.7:
            self.goal_hierarchy.activate_goal("avoid_immediate_collision")
    
    def _manage_temporal_options(self, goal_ids: List[str]):
        """Manage selection and execution of temporal options"""
        # Continue current option if active
        if self.temporal_options.active_option:
            if not self.temporal_options.continue_option():
                # Option terminated, select new one
                self._select_new_option(goal_ids)
        else:
            # No active option, select one
            self._select_new_option(goal_ids)
    
    def _select_new_option(self, goal_ids: List[str]):
        """Select a new temporal option based on active goals"""
        for goal_id in goal_ids:
            option = self.temporal_options.select_option_for_goal(goal_id)
            if option:
                self.temporal_options.initiate_option(option)
                break
    
    def process_reward(self, reward: float, done: bool):
        """Process reward and update goal completions"""
        hierarchical_reward = reward
        
        # Check goal completions based on reward
        if reward > 0:
            # Positive reward - progress on food acquisition goal
            food_goals = ["move_toward_nearest_food", "efficient_food_collection"]
            for goal_id in food_goals:
                if goal_id in self.goal_hierarchy.goals:
                    goal_reward = self.goal_hierarchy.complete_goal(goal_id, success=True)
                    hierarchical_reward += goal_reward
        
        if done and reward < 0:
            # Death - fail survival goals
            survival_goals = ["avoid_immediate_collision", "long_term_survival"]
            for goal_id in survival_goals:
                if goal_id in self.goal_hierarchy.goals:
                    goal_penalty = self.goal_hierarchy.complete_goal(goal_id, success=False)
                    hierarchical_reward += goal_penalty
        
        # Update performance metrics
        self.performance_metrics['hierarchical_reward'] += hierarchical_reward
        
        return hierarchical_reward
    
    def _estimate_snake_length(self, state: np.ndarray) -> int:
        """Estimate snake length from state vector"""
        # This is a simplified estimation
        return max(3, int(state[5] * 10))  # Assuming index 5 has length info
    
    def _estimate_collision_risk(self, state: np.ndarray) -> float:
        """Estimate collision risk from danger detection in state"""
        # Assume first 3 elements are danger detection
        return max(state[0], state[1], state[2]) if len(state) > 3 else 0.0
    
    def _estimate_food_distance(self, state: np.ndarray) -> float:
        """Estimate distance to nearest food"""
        # Assume index 22 has food distance info
        return state[22] if len(state) > 22 else 1.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        active_goals = len(self.goal_hierarchy.active_goals)
        completed_goals = len([g for g in self.goal_hierarchy.goals.values() if g.progress >= 1.0])
        
        avg_option_success = np.mean([opt.success_rate for opt in self.temporal_options.options.values()])
        
        return {
            'active_goals': active_goals,
            'completed_goals': completed_goals,
            'total_goals': len(self.goal_hierarchy.goals),
            'average_option_success_rate': avg_option_success,
            'step_count': self.step_count,
            'hierarchical_reward': self.performance_metrics['hierarchical_reward']
        }

if __name__ == "__main__":
    # Test the HRM system
    print("🧠 Testing Hierarchical Reasoning Model")
    
    hrm = HierarchicalReasoningModel()
    
    # Simulate some game states
    test_state = np.random.rand(33)
    
    for step in range(10):
        action, explanation = hrm.select_action(test_state, epsilon=0.1)
        reward = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        done = reward < 0
        
        hierarchical_reward = hrm.process_reward(reward, done)
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, H-Reward={hierarchical_reward:.2f}")
        print(f"  Active Goals: {explanation['active_goals']}")
        print(f"  Active Option: {explanation['active_option']}")
        
        if done:
            break
    
    print("\n📊 Final Metrics:")
    metrics = hrm.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")