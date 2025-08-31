"""
HRM-Enhanced DQN - Combining Hierarchical Reasoning with Deep Q-Learning
Integrates the HRM system with Enhanced DQN for superior strategic gameplay
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from enhanced_dqn import EnhancedDQN, ReplayInformedTrainer
from hrm_system import HierarchicalReasoningModel, GoalType, OptionStatus

class HRMEnhancedDQN(nn.Module):
    """
    Enhanced DQN with integrated Hierarchical Reasoning Model
    Combines replay embeddings with goal-conditional hierarchical planning
    """
    
    def __init__(self, input_size: int = 33, hidden_size: int = 256, output_size: int = 3,
                 use_replay_embeddings: bool = True, replay_embedding_size: int = 32):
        super(HRMEnhancedDQN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_replay_embeddings = use_replay_embeddings
        self.replay_embedding_size = replay_embedding_size
        
        # Initialize Enhanced DQN components
        self.enhanced_dqn = EnhancedDQN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_replay_embeddings=use_replay_embeddings,
            replay_embedding_size=replay_embedding_size
        )
        
        # Initialize HRM system
        self.hrm = HierarchicalReasoningModel(state_size=input_size)
        
        # HRM-DQN integration layers
        self.hrm_integration = nn.Sequential(
            nn.Linear(hidden_size + 16, hidden_size),  # 16 is goal embedding size from HRM
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Hierarchical value decomposition
        self.meta_value_stream = nn.Linear(hidden_size, 1)      # Meta-level value
        self.strategic_value_stream = nn.Linear(hidden_size, 1) # Strategic-level value
        self.tactical_value_stream = nn.Linear(hidden_size, 1)  # Tactical-level value
        
        # Goal-conditioned advantage streams
        self.survival_advantage = nn.Linear(hidden_size, output_size)
        self.food_advantage = nn.Linear(hidden_size, output_size)
        self.exploration_advantage = nn.Linear(hidden_size, output_size)
        
        # Temporal option integration
        self.option_selector = nn.Linear(hidden_size, 6)  # 6 temporal options
        
        # Strategic projection layer for dimension matching (initialize as None, create when needed)
        self.strategic_projection = None
        
        print("🧠 HRM-Enhanced DQN initialized with hierarchical reasoning capabilities")
    
    def forward(self, x: torch.Tensor, use_hrm: bool = True, 
                replay_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with integrated HRM reasoning
        
        Args:
            x: Current state tensor
            use_hrm: Whether to use HRM for decision making
            replay_context: Pre-computed replay embeddings
            
        Returns:
            Dictionary containing Q-values, HRM outputs, and hierarchical values
        """
        batch_size = x.size(0)
        
        # Get Enhanced DQN base features
        if hasattr(self.enhanced_dqn, 'state_encoder'):
            state_features = self.enhanced_dqn.state_encoder(x)
        else:
            # Fallback for compatibility
            state_features = F.relu(torch.randn(batch_size, self.hidden_size).to(x.device))
        
        # Integrate replay context if available
        if self.use_replay_embeddings and replay_context is not None:
            if replay_context.size(0) != batch_size:
                if replay_context.size(0) < batch_size:
                    repeat_factor = (batch_size + replay_context.size(0) - 1) // replay_context.size(0)
                    replay_context = replay_context.repeat(repeat_factor, 1)[:batch_size]
                else:
                    replay_context = replay_context[:batch_size]
            
            combined_features = torch.cat([state_features, replay_context], dim=1)
            if hasattr(self.enhanced_dqn, 'fusion_layer'):
                fused_features = self.enhanced_dqn.fusion_layer(combined_features)
            else:
                fused_features = state_features
        else:
            fused_features = state_features
        
        # Get HRM reasoning if enabled
        hrm_outputs = {}
        if use_hrm:
            # Convert tensor to numpy for HRM (single sample for now)
            if batch_size == 1:
                state_np = x.cpu().numpy().flatten()
                action, hrm_explanation = self.hrm.select_action(state_np, epsilon=0.0)
                
                # Convert HRM outputs to tensors
                active_goals = hrm_explanation.get('active_goals', [])
                goal_priorities = hrm_explanation.get('goal_priorities', [])
                
                # Create goal context vector
                goal_context = self._create_goal_context(active_goals, goal_priorities, x.device)
                
                # Integrate HRM context
                hrm_integrated_features = self.hrm_integration(
                    torch.cat([fused_features, goal_context], dim=1)
                )
                
                hrm_outputs = {
                    'hrm_action': action,
                    'active_goals': active_goals,
                    'goal_priorities': goal_priorities,
                    'goal_values': hrm_explanation.get('goal_values', {}),
                    'active_option': hrm_explanation.get('active_option'),
                    'hierarchical_reasoning': True
                }
            else:
                # For batch processing, use standard features
                hrm_integrated_features = fused_features
                hrm_outputs['hierarchical_reasoning'] = False
        else:
            hrm_integrated_features = fused_features
            hrm_outputs['hierarchical_reasoning'] = False
        
        # Apply strategic layers from Enhanced DQN
        if hasattr(self.enhanced_dqn, 'strategic_layers'):
            strategic_features = self.enhanced_dqn.strategic_layers(hrm_integrated_features)
            # Strategic layers output hidden_size//2, need to match for HRM layers
            if strategic_features.size(-1) != self.hidden_size:
                # Create strategic projection layer if needed
                if self.strategic_projection is None:
                    self.strategic_projection = nn.Linear(strategic_features.size(-1), self.hidden_size).to(strategic_features.device)
                strategic_features = self.strategic_projection(strategic_features)
        else:
            strategic_features = hrm_integrated_features
        
        # Hierarchical value decomposition
        meta_value = self.meta_value_stream(strategic_features)
        strategic_value = self.strategic_value_stream(strategic_features)
        tactical_value = self.tactical_value_stream(strategic_features)
        
        # Goal-conditioned advantage streams
        survival_advantage = self.survival_advantage(strategic_features)
        food_advantage = self.food_advantage(strategic_features)
        exploration_advantage = self.exploration_advantage(strategic_features)
        
        # Combine advantages based on active goals
        if hrm_outputs.get('active_goals'):
            goal_weights = self._calculate_goal_weights(
                hrm_outputs['active_goals'], 
                hrm_outputs.get('goal_priorities', [])
            )
            
            weighted_advantage = (
                goal_weights.get('survival', 0.33) * survival_advantage +
                goal_weights.get('food', 0.33) * food_advantage +
                goal_weights.get('exploration', 0.34) * exploration_advantage
            )
        else:
            # Default weighting
            weighted_advantage = (
                0.5 * survival_advantage + 
                0.4 * food_advantage + 
                0.1 * exploration_advantage
            )
        
        # Hierarchical Q-values combining all levels
        hierarchical_value = 0.5 * meta_value + 0.3 * strategic_value + 0.2 * tactical_value
        
        # Final Q-values using hierarchical dueling architecture
        q_values = hierarchical_value + (weighted_advantage - weighted_advantage.mean(dim=1, keepdim=True))
        
        # Temporal option selection
        option_logits = self.option_selector(strategic_features)
        
        return {
            'q_values': q_values,
            'meta_value': meta_value,
            'strategic_value': strategic_value,
            'tactical_value': tactical_value,
            'survival_advantage': survival_advantage,
            'food_advantage': food_advantage,
            'exploration_advantage': exploration_advantage,
            'option_logits': option_logits,
            'hrm_outputs': hrm_outputs
        }
    
    def _create_goal_context(self, active_goals: List[str], goal_priorities: List[float], 
                           device: torch.device) -> torch.Tensor:
        """Create goal context vector for integration"""
        # Simple goal encoding (16-dimensional)
        context = torch.zeros(1, 16).to(device)
        
        goal_mappings = {
            'maximize_score': 0,
            'long_term_survival': 1,
            'efficient_food_collection': 2,
            'territory_control': 3,
            'avoid_immediate_collision': 4,
            'move_toward_nearest_food': 5,
            'maintain_safe_distance': 6,
            'optimize_path_efficiency': 7,
            'explore_unknown_areas': 8
        }
        
        for i, goal in enumerate(active_goals[:8]):  # Top 8 goals max
            if goal in goal_mappings:
                idx = goal_mappings[goal]
                priority = goal_priorities[i] if i < len(goal_priorities) else 1.0
                context[0, idx] = priority
        
        return context
    
    def _calculate_goal_weights(self, active_goals: List[str], 
                              goal_priorities: List[float]) -> Dict[str, float]:
        """Calculate weights for different goal categories"""
        weights = {'survival': 0.0, 'food': 0.0, 'exploration': 0.0}
        total_priority = sum(goal_priorities) if goal_priorities else 1.0
        
        for i, goal in enumerate(active_goals):
            priority = goal_priorities[i] if i < len(goal_priorities) else 1.0
            normalized_priority = priority / total_priority
            
            if any(keyword in goal for keyword in ['survival', 'collision', 'safe']):
                weights['survival'] += normalized_priority
            elif any(keyword in goal for keyword in ['food', 'collection', 'acquisition']):
                weights['food'] += normalized_priority
            elif any(keyword in goal for keyword in ['explore', 'territory', 'control']):
                weights['exploration'] += normalized_priority
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        else:
            weights = {'survival': 0.5, 'food': 0.4, 'exploration': 0.1}
        
        return weights
    
    def get_action_with_hrm_explanation(self, state: torch.Tensor, 
                                      epsilon: float = 0.0) -> Tuple[int, Dict[str, Any]]:
        """
        Get action with comprehensive HRM explanation
        """
        with torch.no_grad():
            outputs = self.forward(state.unsqueeze(0) if len(state.shape) == 1 else state, use_hrm=True)
            q_values = outputs['q_values'].squeeze()
            hrm_outputs = outputs['hrm_outputs']
            
            # Epsilon-greedy with HRM consideration
            if np.random.random() < epsilon:
                action = np.random.randint(self.output_size)
                decision_type = "exploration"
            else:
                # Use HRM action if available and confident
                if (hrm_outputs.get('hierarchical_reasoning') and 
                    hrm_outputs.get('hrm_action') is not None):
                    hrm_action = hrm_outputs['hrm_action']
                    dqn_action = int(torch.argmax(q_values).item())
                    
                    # Blend HRM and DQN decisions (prefer HRM for strategic decisions)
                    if abs(hrm_action - dqn_action) <= 1:  # Similar decisions
                        action = hrm_action
                        decision_type = "hrm_aligned"
                    else:
                        # Use Q-value confidence to decide
                        q_confidence = torch.max(q_values) - torch.mean(q_values)
                        if q_confidence > 1.0:  # High DQN confidence
                            action = dqn_action
                            decision_type = "dqn_override"
                        else:
                            action = hrm_action
                            decision_type = "hrm_strategic"
                else:
                    action = int(torch.argmax(q_values).item())
                    decision_type = "dqn_only"
            
            # Comprehensive explanation
            explanation = {
                'decision_type': decision_type,
                'q_values': q_values.cpu().numpy().tolist(),
                'selected_action': action,
                'dqn_confidence': float(torch.max(q_values) - torch.mean(q_values)),
                'hierarchical_values': {
                    'meta': float(outputs['meta_value'].item()),
                    'strategic': float(outputs['strategic_value'].item()),
                    'tactical': float(outputs['tactical_value'].item())
                },
                'goal_advantages': {
                    'survival': float(torch.max(outputs['survival_advantage']).item()),
                    'food': float(torch.max(outputs['food_advantage']).item()),
                    'exploration': float(torch.max(outputs['exploration_advantage']).item())
                },
                **hrm_outputs  # Include all HRM outputs
            }
            
            return action, explanation
    
    def update_hrm_rewards(self, reward: float, done: bool) -> float:
        """Update HRM system with rewards and get hierarchical reward"""
        return self.hrm.process_reward(reward, done)
    
    def get_hrm_metrics(self) -> Dict[str, Any]:
        """Get HRM performance metrics"""
        return self.hrm.get_performance_metrics()

class HRMReplayInformedTrainer(ReplayInformedTrainer):
    """Extended trainer for HRM-Enhanced DQN"""
    
    def __init__(self, hrm_enhanced_dqn: HRMEnhancedDQN, learning_rate: float = 0.001):
        self.hrm_model = hrm_enhanced_dqn
        # Initialize parent with the enhanced_dqn component
        super().__init__(hrm_enhanced_dqn.enhanced_dqn, learning_rate)
        
        # Override optimizer to include HRM parameters
        self.optimizer = torch.optim.Adam(hrm_enhanced_dqn.parameters(), lr=learning_rate)
        
        # HRM-specific training components
        self.hierarchical_loss_weight = 0.3
        self.goal_value_loss_weight = 0.2
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, next_states: torch.Tensor,
                   dones: torch.Tensor, gamma: float = 0.99) -> Dict[str, float]:
        """
        Enhanced training step with HRM components
        """
        batch_size = states.size(0)
        
        # Get current outputs
        current_outputs = self.hrm_model.forward(states, use_hrm=False)  # Disable HRM for batch training
        current_q_values = current_outputs['q_values'].gather(1, actions.unsqueeze(1))
        
        # Get target Q-values
        with torch.no_grad():
            next_outputs = self.hrm_model.forward(next_states, use_hrm=False)
            next_q_values = next_outputs['q_values']
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (gamma * max_next_q_values * ~dones)
        
        # Standard DQN loss
        dqn_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Hierarchical value consistency loss
        meta_values = current_outputs['meta_value'].squeeze()
        strategic_values = current_outputs['strategic_value'].squeeze()
        tactical_values = current_outputs['tactical_value'].squeeze()
        
        # Ensure hierarchical consistency (meta >= strategic >= tactical)
        hierarchical_loss = (
            F.relu(strategic_values - meta_values).mean() +
            F.relu(tactical_values - strategic_values).mean()
        )
        
        # Goal-conditioned advantage consistency
        survival_adv = current_outputs['survival_advantage']
        food_adv = current_outputs['food_advantage']
        exploration_adv = current_outputs['exploration_advantage']
        
        # Advantage values should sum to zero (dueling architecture property)
        advantage_consistency_loss = (
            survival_adv.mean(1).pow(2).mean() +
            food_adv.mean(1).pow(2).mean() +
            exploration_adv.mean(1).pow(2).mean()
        )
        
        # Combined loss
        total_loss = (
            dqn_loss +
            self.hierarchical_loss_weight * hierarchical_loss +
            self.goal_value_loss_weight * advantage_consistency_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.hrm_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'dqn_loss': dqn_loss.item(),
            'hierarchical_loss': hierarchical_loss.item(),
            'advantage_consistency_loss': advantage_consistency_loss.item()
        }

def create_hrm_enhanced_dqn(input_size: int = 33, hidden_size: int = 256, 
                           output_size: int = 3) -> HRMEnhancedDQN:
    """Factory function to create HRM-Enhanced DQN"""
    return HRMEnhancedDQN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        use_replay_embeddings=True,
        replay_embedding_size=32
    )

if __name__ == "__main__":
    # Test the HRM-Enhanced DQN
    print("🧠 Testing HRM-Enhanced DQN")
    
    model = create_hrm_enhanced_dqn()
    trainer = HRMReplayInformedTrainer(model)
    
    # Test forward pass
    batch_size = 4
    state_size = 33
    test_states = torch.randn(batch_size, state_size)
    
    print(f"🔍 Testing forward pass with batch size {batch_size}")
    outputs = model(test_states, use_hrm=False)  # Disable HRM for batch test
    print(f"✅ Q-values shape: {outputs['q_values'].shape}")
    print(f"🏗️ Hierarchical values: Meta={outputs['meta_value'].shape}, Strategic={outputs['strategic_value'].shape}, Tactical={outputs['tactical_value'].shape}")
    
    # Test single-state HRM reasoning
    single_state = torch.randn(state_size)
    action, explanation = model.get_action_with_hrm_explanation(single_state)
    print(f"\n🎯 HRM Action Selection:")
    print(f"   Selected action: {action}")
    print(f"   Decision type: {explanation['decision_type']}")
    print(f"   Active goals: {explanation.get('active_goals', [])}")
    print(f"   HRM reasoning: {explanation.get('hierarchical_reasoning', False)}")
    
    print("\n🏆 HRM-Enhanced DQN testing completed successfully!")