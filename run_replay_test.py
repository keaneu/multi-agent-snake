#!/usr/bin/env python3
"""
Test script to run and validate the replay sample
"""
import json
import time
from replay_agent import ReplayAgent, ReplayEpisode, ReplayFrame

def test_replay_execution():
    """Test running the actual replay sample"""
    print("🎥 Testing Replay Sample Execution")
    print("=" * 60)
    
    replay_file = "replays/episode_1724006327_score_74.json"
    
    try:
        # Load replay data
        with open(replay_file, 'r') as f:
            replay_data = json.load(f)
        
        print(f"📂 Loaded: {replay_data['episode_id']}")
        print(f"🎯 Target Score: {replay_data['final_score']}")
        print(f"🎬 Frame Count: {len(replay_data['frames'])}")
        
        # Create replay episode object
        episode = ReplayEpisode(
            episode_id=replay_data["episode_id"],
            final_score=replay_data["final_score"],
            frames=[ReplayFrame(**frame) for frame in replay_data["frames"]]
        )
        
        print(f"\n🎮 SIMULATING REPLAY EXECUTION")
        print("-" * 60)
        
        # Simulate replay execution frame by frame
        max_score_a = 0
        max_score_b = 0
        foods_eaten_a = 0
        foods_eaten_b = 0
        
        for i, frame in enumerate(episode.frames):
            # Track scores
            score_a = frame.scores.get('A', 0)
            score_b = frame.scores.get('B', 0)
            
            max_score_a = max(max_score_a, score_a)
            max_score_b = max(max_score_b, score_b)
            
            # Track food consumption (when snake length increases)
            if i > 0:
                prev_frame = episode.frames[i-1]
                if len(frame.snakes['A']) > len(prev_frame.snakes['A']):
                    foods_eaten_a += 1
                    print(f"   🍎 Frame {i+1}: Snake A ate food! Score: {score_a}")
                if len(frame.snakes['B']) > len(prev_frame.snakes['B']):
                    foods_eaten_b += 1
                    print(f"   🍎 Frame {i+1}: Snake B ate food! Score: {score_b}")
            
            # Show progress every 5 frames
            if (i + 1) % 5 == 0 or i == len(episode.frames) - 1:
                print(f"Frame {i+1:2d}: A={score_a:2d}pts ({len(frame.snakes['A'])} segs), "
                      f"B={score_b:2d}pts ({len(frame.snakes['B'])} segs)")
        
        print(f"\n📈 REPLAY EXECUTION RESULTS")
        print("-" * 60)
        print(f"🥇 Snake A Final Score: {max_score_a} points")
        print(f"🥈 Snake B Final Score: {max_score_b} points")
        print(f"🍎 Snake A Foods Eaten: {foods_eaten_a}")
        print(f"🍎 Snake B Foods Eaten: {foods_eaten_b}")
        print(f"🏆 Claimed Final Score: {episode.final_score}")
        
        # Validation
        actual_max = max(max_score_a, max_score_b)
        if actual_max == episode.final_score:
            print(f"✅ Score Validation: PASSED ({actual_max} points)")
            validation_passed = True
        else:
            print(f"❌ Score Validation: FAILED (Expected: {episode.final_score}, Got: {actual_max})")
            validation_passed = False
        
        # Test replay agent loading
        print(f"\n🎭 TESTING REPLAY AGENT")
        print("-" * 60)
        
        agent = ReplayAgent(high_score_threshold=50)
        print(f"✅ ReplayAgent created (threshold: {agent.high_score_threshold})")
        
        # Test if this score would trigger saving
        if actual_max > agent.high_score_threshold:
            print(f"✅ Score {actual_max} exceeds threshold {agent.high_score_threshold} - would be saved")
        else:
            print(f"⚠️  Score {actual_max} below threshold {agent.high_score_threshold} - would not be saved")
        
        return validation_passed
        
    except Exception as e:
        print(f"❌ Error during replay execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_replay_file_integrity():
    """Test replay file structure and data integrity"""
    print(f"\n🔍 TESTING REPLAY FILE INTEGRITY")
    print("-" * 60)
    
    replay_file = "replays/episode_1724006327_score_74.json"
    
    try:
        with open(replay_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['episode_id', 'final_score', 'frames']
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field: {field}")
                return False
            else:
                print(f"✅ Field present: {field}")
        
        # Check frames structure
        if not data['frames']:
            print("❌ No frames in replay")
            return False
        
        frame_count = len(data['frames'])
        print(f"✅ Frame count: {frame_count}")
        
        # Check first frame structure
        first_frame = data['frames'][0]
        required_frame_fields = ['snakes', 'food', 'obstacles', 'scores', 'game_phase']
        for field in required_frame_fields:
            if field not in first_frame:
                print(f"❌ Missing frame field: {field}")
                return False
        
        print(f"✅ Frame structure valid")
        
        # Check snakes data
        if 'A' not in first_frame['snakes'] or 'B' not in first_frame['snakes']:
            print("❌ Missing snake data")
            return False
        
        print(f"✅ Snake data present for both agents")
        print(f"   Snake A segments: {len(first_frame['snakes']['A'])}")
        print(f"   Snake B segments: {len(first_frame['snakes']['B'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking file integrity: {e}")
        return False

if __name__ == "__main__":
    print("🎮 Multi-Agent Snake Replay Validation Test")
    print("=" * 60)
    
    # Run tests
    integrity_passed = test_replay_file_integrity()
    execution_passed = test_replay_execution()
    
    print(f"\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    
    if integrity_passed and execution_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Replay sample is valid and ready for use")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        if not integrity_passed:
            print("   - File integrity check failed")
        if not execution_passed:
            print("   - Replay execution validation failed")
        exit(1)