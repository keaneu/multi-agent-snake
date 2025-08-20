#!/usr/bin/env python3
"""
🎯 30+ Point Training Plan
Comprehensive strategy to achieve consistent 30+ point scores
"""

from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class TrainingPhase:
    """Individual phase of the 30+ training plan"""
    name: str
    duration_minutes: int
    score_threshold: int
    reward_multipliers: Dict[str, float]
    curriculum_focus: str
    success_criteria: str

class ThirtyPlusTrainingPlan:
    """
    Progressive training plan to achieve consistent 30+ point scores
    Based on analysis of successful 30+ point episodes and dense reward system
    """
    
    def __init__(self):
        self.phases = self._design_training_phases()
        self.enhanced_rewards = self._design_30plus_rewards()
        
    def _design_training_phases(self) -> List[TrainingPhase]:
        """Design progressive training phases targeting 30+ points"""
        
        return [
            # Phase 1: Foundation Building (15-25 points)
            TrainingPhase(
                name="Foundation Building",
                duration_minutes=20,
                score_threshold=15,
                reward_multipliers={
                    "food_proximity_reward": 1.0,
                    "wall_avoidance_reward": 1.2,
                    "efficient_movement_reward": 0.8,
                    "safe_exploration_reward": 1.0,
                    "strategic_positioning_reward": 1.0
                },
                curriculum_focus="Basic survival and food acquisition",
                success_criteria="Consistently achieve 15+ points, 80% episodes with at least 1 food"
            ),
            
            # Phase 2: Intermediate Mastery (20-30 points)
            TrainingPhase(
                name="Intermediate Mastery", 
                duration_minutes=25,
                score_threshold=25,
                reward_multipliers={
                    "food_proximity_reward": 1.5,
                    "wall_avoidance_reward": 1.0,
                    "efficient_movement_reward": 1.2,
                    "safe_exploration_reward": 1.3,
                    "strategic_positioning_reward": 1.4
                },
                curriculum_focus="Multi-food acquisition and efficient navigation",
                success_criteria="Consistently achieve 25+ points, 60% episodes with 2+ foods"
            ),
            
            # Phase 3: Advanced Strategy (25-35 points)
            TrainingPhase(
                name="Advanced Strategy",
                duration_minutes=30,
                score_threshold=30,
                reward_multipliers={
                    "food_proximity_reward": 2.0,
                    "wall_avoidance_reward": 0.8,
                    "efficient_movement_reward": 1.5,
                    "safe_exploration_reward": 1.5,
                    "strategic_positioning_reward": 2.0
                },
                curriculum_focus="Long-term planning and competitive positioning",
                success_criteria="Consistently achieve 30+ points, 40% episodes with 3+ foods"
            ),
            
            # Phase 4: Elite Performance (30+ points)
            TrainingPhase(
                name="Elite Performance",
                duration_minutes=35,
                score_threshold=35,
                reward_multipliers={
                    "food_proximity_reward": 2.5,
                    "wall_avoidance_reward": 1.0,
                    "efficient_movement_reward": 2.0,
                    "safe_exploration_reward": 2.0,
                    "strategic_positioning_reward": 2.5
                },
                curriculum_focus="Consistent 30+ performance and strategic dominance",
                success_criteria="Consistently achieve 35+ points, maintain 30+ minimum"
            ),
            
            # Phase 5: Mastery Consolidation
            TrainingPhase(
                name="Mastery Consolidation",
                duration_minutes=40,
                score_threshold=30,
                reward_multipliers={
                    "food_proximity_reward": 2.0,
                    "wall_avoidance_reward": 1.2,
                    "efficient_movement_reward": 1.8,
                    "safe_exploration_reward": 1.8,
                    "strategic_positioning_reward": 2.2
                },
                curriculum_focus="Consistent 30+ point performance stability",
                success_criteria="95% of episodes achieve 30+ points"
            ),
            
            # Phase 6: Super Elite Performance (60+ points)
            TrainingPhase(
                name="Super Elite Performance",
                duration_minutes=45,
                score_threshold=50,
                reward_multipliers={
                    "food_proximity_reward": 3.0,
                    "wall_avoidance_reward": 0.8,
                    "efficient_movement_reward": 2.5,
                    "safe_exploration_reward": 2.5,
                    "strategic_positioning_reward": 3.0
                },
                curriculum_focus="Breakthrough 60+ point performance with ultra-advanced strategies",
                success_criteria="Consistently achieve 50+ points, target 60+ breakthrough episodes"
            )
        ]
    
    def _design_30plus_rewards(self) -> Dict[str, float]:
        """Enhanced reward configuration optimized for 30+ point performance"""
        
        return {
            # Core rewards enhanced for 30+ performance
            "food_reward": 15.0,  # Increased from 10.0
            "death_penalty": -15.0,  # Increased penalty to discourage risky play
            "step_penalty": -0.005,  # Reduced to encourage longer games
            "survival_reward": 0.2,  # Increased to reward longevity
            
            # Enhanced intermediate rewards for 30+ targets
            "food_proximity_reward": 1.0,  # Doubled from 0.5
            "wall_avoidance_reward": 0.5,  # Increased from 0.3
            "efficient_movement_reward": 0.4,  # Doubled from 0.2
            "length_progression_reward": 2.0,  # Doubled from 1.0
            "safe_exploration_reward": 0.3,  # Doubled from 0.15
            "strategic_positioning_reward": 0.5,  # Doubled from 0.25
            
            # New advanced rewards for 30+ performance
            "multi_food_bonus": 5.0,  # Bonus for 2+ foods in succession
            "efficiency_streak_bonus": 3.0,  # Bonus for efficient movement patterns
            "longevity_bonus": 0.1,  # Per-step bonus after 200 steps
            "dominance_bonus": 2.0,  # Bonus for being ahead of opponent
        }
    
    def get_phase_config(self, phase_index: int) -> Dict:
        """Get training configuration for specific phase"""
        if phase_index >= len(self.phases):
            phase_index = len(self.phases) - 1  # Use final phase
            
        phase = self.phases[phase_index]
        
        # Create enhanced config for this phase
        enhanced_rewards = self.enhanced_rewards.copy()
        
        # Apply phase-specific multipliers
        for reward_key, multiplier in phase.reward_multipliers.items():
            if reward_key in enhanced_rewards:
                enhanced_rewards[reward_key] *= multiplier
        
        return {
            "phase": phase,
            "rewards": enhanced_rewards,
            "training_duration": phase.duration_minutes * 60,  # Convert to seconds
            "replay_threshold": phase.score_threshold,
            "epsilon_decay": 0.9995,  # Slower decay for longer training
            "learning_rate": 0.0008,  # Slightly reduced for stability
            "batch_size": 64,  # Increased for better learning
        }
    
    def print_training_plan(self):
        """Print comprehensive training plan overview"""
        print("🎯 30+ POINT TRAINING PLAN")
        print("=" * 50)
        
        total_duration = sum(phase.duration_minutes for phase in self.phases)
        print(f"📊 Total Training Duration: {total_duration} minutes ({total_duration/60:.1f} hours)")
        print(f"🎯 Target: Consistent 30+ point performance")
        print(f"📈 Phases: {len(self.phases)} progressive stages")
        print()
        
        for i, phase in enumerate(self.phases, 1):
            print(f"📋 Phase {i}: {phase.name}")
            print(f"   ⏱️  Duration: {phase.duration_minutes} minutes")
            print(f"   🎯 Score Target: {phase.score_threshold}+")
            print(f"   🔍 Focus: {phase.curriculum_focus}")
            print(f"   ✅ Success: {phase.success_criteria}")
            print()
        
        print("🚀 ENHANCED REWARDS FOR 30+ PERFORMANCE:")
        print("-" * 40)
        for key, value in self.enhanced_rewards.items():
            print(f"   {key}: {value}")

def generate_training_script(plan: ThirtyPlusTrainingPlan) -> str:
    """Generate executable training script"""
    
    script = '''#!/bin/bash
# 🎯 30+ Point Training Campaign
# Generated training script for consistent 30+ point performance

echo "🎯 Starting 30+ Point Training Campaign"
echo "======================================"

'''
    
    for i, phase in enumerate(plan.phases):
        script += f'''
echo "📋 Phase {i+1}: {phase.name}"
echo "⏱️  Duration: {phase.duration_minutes} minutes"
echo "🎯 Target: {phase.score_threshold}+ points"
echo "🔍 Focus: {phase.curriculum_focus}"
echo ""

# Update reward configuration for this phase
python -c "
from ai_training_agent import TrainingConfig
import json

# Apply phase {i+1} reward configuration
config = TrainingConfig()
{_generate_reward_updates(plan.get_phase_config(i)['rewards'])}

print('✅ Phase {i+1} reward configuration applied')
"

# Run training for this phase
echo "🚀 Starting Phase {i+1} training..."
python run_multi_agent_snake.py {phase.duration_minutes * 60}

echo "✅ Phase {i+1} completed"
echo ""
'''
    
    script += '''
echo "🎉 30+ Point Training Campaign Complete!"
echo "📊 Analyzing final performance..."

# Generate performance report
python -c "
import glob
import json

episodes = []
for file in glob.glob('*.json'):
    if 'episode_' in file and 'score_' in file:
        try:
            score = int(file.split('score_')[1].split('.')[0])
            if score >= 30:
                episodes.append(score)
        except:
            pass

episodes.sort(reverse=True)
if episodes:
    avg_30plus = sum(episodes) / len(episodes)
    print(f'🎯 30+ Point Episodes: {len(episodes)}')
    print(f'📈 Average 30+ Score: {avg_30plus:.1f}')
    print(f'🏆 Highest Score: {max(episodes)}')
    success_rate = len(episodes) / len(glob.glob('episode_*.json')) * 100
    print(f'✅ 30+ Success Rate: {success_rate:.1f}%')
else:
    print('⚠️  No 30+ point episodes achieved yet')
"
'''
    
    return script

def _generate_reward_updates(rewards: Dict[str, float]) -> str:
    """Generate Python code to update reward configuration"""
    updates = []
    for key, value in rewards.items():
        updates.append(f"config.{key} = {value}")
    return "\\n".join(updates)

if __name__ == "__main__":
    # Create and display the training plan
    plan = ThirtyPlusTrainingPlan()
    plan.print_training_plan()
    
    print("\\n🔧 IMPLEMENTATION OPTIONS:")
    print("1. Manual phase execution with reward tuning")
    print("2. Automated training script generation")
    print("3. Interactive curriculum progression")
    
    # Generate training script
    training_script = generate_training_script(plan)
    
    with open("train_30plus.sh", "w") as f:
        f.write(training_script)
    
    print("\\n✅ Training plan complete!")
    print("📝 Automated script saved to: train_30plus.sh")
    print("🚀 Ready to begin 30+ point training campaign!")