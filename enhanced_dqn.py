"""
Enhanced DQN with High-Score Replay Embeddings
Integrates champion gameplay patterns via RNN embeddings for superior AI performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from replay_embedding_system import ReplayRNNEmbedding, HighScoreReplayManager

class EnhancedDQN(nn.Module):
    """
    Enhanced Deep Q-Network that incorporates high-score replay embeddings
    for improved strategic decision making
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 use_replay_embeddings: bool = True, replay_embedding_size: int = 32):
        super(EnhancedDQN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_replay_embeddings = use_replay_embeddings
        self.replay_embedding_size = replay_embedding_size
        
        # Current state processing layers
        self.state_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Replay embedding system
        if self.use_replay_embeddings:
            self.replay_embedding_net = ReplayRNNEmbedding(
                input_size=input_size,
                hidden_size=64,
                embedding_size=replay_embedding_size,
                num_layers=2
            )
            
            # Attention mechanism to combine current state with replay patterns
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            
            # Combined feature processing
            combined_size = hidden_size + replay_embedding_size
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        else:
            combined_size = hidden_size
            self.fusion_layer = nn.Identity()
        
        # Strategic reasoning layers
        self.strategic_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Action value output layers
        self.value_stream = nn.Linear(hidden_size // 2, 1)
        self.advantage_stream = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Replay manager for getting embeddings - Elite 40-600 Training (all 40+ score patterns for comprehensive learning)
        if self.use_replay_embeddings:
            self.replay_manager = HighScoreReplayManager(min_score_threshold=40, max_score_threshold=600)
            print(f"🎯 Enhanced DQN initialized with {len(self.replay_manager.replay_patterns)} replay patterns (40-600 score range)")
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
    
    def get_replay_context(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get replay embedding context for current batch"""
        if not self.use_replay_embeddings or not hasattr(self, 'replay_manager'):
            return None
        
        try:
            # Get replay sequences and pattern types
            sequences, pattern_types = self.replay_manager.get_replay_embeddings(batch_size)
            
            if sequences is not None and pattern_types is not None:
                # Generate embeddings
                with torch.no_grad():  # Don't update replay embedding weights during main training
                    replay_embeddings = self.replay_embedding_net(sequences, pattern_types)
                return replay_embeddings
        except Exception as e:
            print(f"⚠️  Error getting replay context: {e}")
        
        return None
    
    def forward(self, x: torch.Tensor, replay_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional replay embedding integration
        
        Args:
            x: Current state tensor (batch_size, input_size)
            replay_context: Pre-computed replay embeddings (batch_size, replay_embedding_size)
        
        Returns:
            Q-values for each action (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Process current state
        state_features = self.state_encoder(x)
        
        # Integrate replay embeddings if available
        if self.use_replay_embeddings:
            if replay_context is None:
                replay_context = self.get_replay_context(batch_size)
            
            if replay_context is not None:
                # Ensure replay context matches batch size
                if replay_context.size(0) != batch_size:
                    # Repeat or truncate to match batch size
                    if replay_context.size(0) < batch_size:
                        repeat_factor = (batch_size + replay_context.size(0) - 1) // replay_context.size(0)
                        replay_context = replay_context.repeat(repeat_factor, 1)[:batch_size]
                    else:
                        replay_context = replay_context[:batch_size]
                
                # Combine state features with replay embeddings
                combined_features = torch.cat([state_features, replay_context], dim=1)
                fused_features = self.fusion_layer(combined_features)
            else:
                # No replay context available, use state features only
                fused_features = state_features
        else:
            fused_features = state_features
        
        # Strategic reasoning
        strategic_features = self.strategic_layers(fused_features)
        
        # Dueling DQN architecture
        value = self.value_stream(strategic_features)
        advantage = self.advantage_stream(strategic_features)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action_with_explanation(self, state: torch.Tensor, epsilon: float = 0.0) -> Tuple[int, dict]:
        """
        Get action with explanation of decision process
        
        Args:
            state: Current game state
            epsilon: Exploration rate
        
        Returns:
            action: Selected action
            explanation: Dict with decision reasoning
        """
        with torch.no_grad():
            # Get Q-values
            q_values = self.forward(state.unsqueeze(0))
            q_values_np = q_values.squeeze().cpu().numpy()
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(self.output_size)
                decision_type = "exploration"
            else:
                action = int(np.argmax(q_values_np))
                decision_type = "exploitation"
            
            # Create explanation
            explanation = {
                'decision_type': decision_type,
                'q_values': q_values_np.tolist(),
                'selected_action': action,
                'confidence': float(np.max(q_values_np) - np.mean(q_values_np)),
                'replay_informed': self.use_replay_embeddings
            }
            
            return action, explanation

class ReplayInformedTrainer:
    """Training wrapper that manages replay-informed DQN training"""
    
    def __init__(self, enhanced_dqn: EnhancedDQN, learning_rate: float = 0.001):
        self.model = enhanced_dqn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay embedding caching for efficiency
        self.replay_cache = {}
        self.cache_size = 100
    
    def get_cached_replay_context(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get cached replay context or generate new one"""
        if batch_size in self.replay_cache:
            return self.replay_cache[batch_size]
        
        # Generate new context
        context = self.model.get_replay_context(batch_size)
        
        if context is not None and len(self.replay_cache) < self.cache_size:
            self.replay_cache[batch_size] = context
        
        return context
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   rewards: torch.Tensor, next_states: torch.Tensor, 
                   dones: torch.Tensor, gamma: float = 0.99) -> float:
        """
        Perform one training step with replay-informed learning
        
        Args:
            states: Current states (batch_size, state_size)
            actions: Actions taken (batch_size,)
            rewards: Rewards received (batch_size,)
            next_states: Next states (batch_size, state_size)
            dones: Episode done flags (batch_size,)
            gamma: Discount factor
        
        Returns:
            loss: Training loss value
        """
        batch_size = states.size(0)
        
        # Get replay context for current batch
        replay_context = self.get_cached_replay_context(batch_size)
        
        # Current Q-values
        current_q_values = self.model(states, replay_context)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values (no replay context for target to reduce overfitting)
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_replay_patterns(self):
        """Update replay patterns from new high-score files"""
        if hasattr(self.model, 'replay_manager'):
            old_count = len(self.model.replay_manager.replay_patterns)
            self.model.replay_manager.load_high_score_patterns()
            new_count = len(self.model.replay_manager.replay_patterns)
            
            if new_count > old_count:
                print(f"🆕 Updated replay patterns: {old_count} → {new_count}")
                # Clear cache to incorporate new patterns
                self.replay_cache.clear()

def create_enhanced_dqn(input_size: int = 33, hidden_size: int = 256, output_size: int = 3) -> EnhancedDQN:
    """Factory function to create an enhanced DQN with optimal settings"""
    return EnhancedDQN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        use_replay_embeddings=True,
        replay_embedding_size=32
    )

if __name__ == "__main__":
    # Test the enhanced DQN
    print("🧠 Testing Enhanced DQN with Replay Embeddings")
    
    # Create model
    model = create_enhanced_dqn()
    trainer = ReplayInformedTrainer(model)
    
    # Test forward pass
    batch_size = 4
    state_size = 33
    test_states = torch.randn(batch_size, state_size)
    
    print(f"🔍 Testing forward pass with batch size {batch_size}")
    q_values = model(test_states)
    print(f"✅ Q-values shape: {q_values.shape}")
    print(f"📊 Sample Q-values: {q_values[0].detach().numpy()}")
    
    # Test action selection with explanation
    single_state = torch.randn(1, state_size)
    action, explanation = model.get_action_with_explanation(single_state.squeeze())
    print(f"🎯 Selected action: {action}")
    print(f"📝 Decision explanation: {explanation}")
    
    print("🏆 Enhanced DQN testing completed successfully!")