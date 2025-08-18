#!/usr/bin/env python3
"""
Play back the 74-point replay with detailed visualization
"""
import json
import time
from replay_agent import ReplayAgent, ReplayEpisode, ReplayFrame

def visualize_game_state(frame, frame_num, total_frames):
    """Visualize a single frame of the game"""
    print(f"\n🎬 FRAME {frame_num}/{total_frames}")
    print("=" * 60)
    
    # Create 20x20 grid
    grid = [['.' for _ in range(20)] for _ in range(20)]
    
    # Place obstacles
    for obs in frame.obstacles:
        x, y = obs
        if 0 <= x < 20 and 0 <= y < 20:
            grid[y][x] = '█'
    
    # Place food
    for food in frame.food:
        x, y = food
        if 0 <= x < 20 and 0 <= y < 20:
            grid[y][x] = '🍎'
    
    # Place Snake A (blue)
    for i, (x, y) in enumerate(frame.snakes['A']):
        if 0 <= x < 20 and 0 <= y < 20:
            if i == 0:  # Head
                grid[y][x] = 'A'
            else:  # Body
                grid[y][x] = 'a'
    
    # Place Snake B (red)
    for i, (x, y) in enumerate(frame.snakes['B']):
        if 0 <= x < 20 and 0 <= y < 20:
            if i == 0:  # Head
                grid[y][x] = 'B'
            else:  # Body
                grid[y][x] = 'b'
    
    # Print grid with coordinates
    print("   " + "".join([f"{i%10}" for i in range(20)]))
    for y, row in enumerate(grid):
        print(f"{y:2d} " + "".join(row))
    
    # Show scores and stats
    score_a = frame.scores.get('A', 0)
    score_b = frame.scores.get('B', 0)
    len_a = len(frame.snakes['A'])
    len_b = len(frame.snakes['B'])
    
    print(f"\n📊 SCORES: Snake A = {score_a:2d} pts ({len_a} segments) | Snake B = {score_b:2d} pts ({len_b} segments)")
    print(f"🍎 FOODS: {len(frame.food)} available | Phase: {frame.game_phase}")

def play_replay_interactive():
    """Play the 74-point replay interactively"""
    print("🎥 MULTI-AGENT SNAKE REPLAY PLAYER")
    print("=" * 60)
    print("Loading 74-point championship episode...")
    
    replay_file = "replays/episode_1724006327_score_74.json"
    
    try:
        with open(replay_file, 'r') as f:
            replay_data = json.load(f)
        
        episode = ReplayEpisode(
            episode_id=replay_data["episode_id"],
            final_score=replay_data["final_score"],
            frames=[ReplayFrame(**frame) for frame in replay_data["frames"]]
        )
        
        print(f"✅ Loaded: {episode.episode_id}")
        print(f"🎯 Final Score: {episode.final_score} points")
        print(f"🎬 Total Frames: {len(episode.frames)}")
        print("\n🎮 LEGEND: A/a = Snake A (head/body), B/b = Snake B (head/body), █ = Obstacle, 🍎 = Food, . = Empty")
        
        print("\n" + "="*60)
        input("Press ENTER to start replay...")
        
        foods_eaten_a = 0
        foods_eaten_b = 0
        
        for i, frame in enumerate(episode.frames):
            visualize_game_state(frame, i+1, len(episode.frames))
            
            # Track food consumption
            if i > 0:
                prev_frame = episode.frames[i-1]
                if len(frame.snakes['A']) > len(prev_frame.snakes['A']):
                    foods_eaten_a += 1
                    print(f"🍎 Snake A ate food! Total foods: {foods_eaten_a}")
                if len(frame.snakes['B']) > len(prev_frame.snakes['B']):
                    foods_eaten_b += 1
                    print(f"🍎 Snake B ate food! Total foods: {foods_eaten_b}")
            
            # Pause between frames (or wait for input)
            if i < len(episode.frames) - 1:  # Don't pause on last frame
                print("\n" + "-" * 60)
                user_input = input("Press ENTER for next frame (or 'q' to quit, 'a' for auto-play): ").strip().lower()
                if user_input == 'q':
                    print("🛑 Replay stopped by user")
                    break
                elif user_input == 'a':
                    print("🎬 Auto-play mode activated...")
                    # Auto-play remaining frames
                    for j in range(i+1, len(episode.frames)):
                        time.sleep(0.5)  # Half second delay
                        frame = episode.frames[j]
                        visualize_game_state(frame, j+1, len(episode.frames))
                        
                        # Track food consumption in auto-play
                        if j > 0:
                            prev_frame = episode.frames[j-1]
                            if len(frame.snakes['A']) > len(prev_frame.snakes['A']):
                                foods_eaten_a += 1
                                print(f"🍎 Snake A ate food! Total foods: {foods_eaten_a}")
                            if len(frame.snakes['B']) > len(prev_frame.snakes['B']):
                                foods_eaten_b += 1
                                print(f"🍎 Snake B ate food! Total foods: {foods_eaten_b}")
                    break
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏁 REPLAY COMPLETED!")
        print("=" * 60)
        final_frame = episode.frames[-1]
        print(f"🥇 FINAL SCORES:")
        print(f"   Snake A: {final_frame.scores.get('A', 0)} points ({len(final_frame.snakes['A'])} segments)")
        print(f"   Snake B: {final_frame.scores.get('B', 0)} points ({len(final_frame.snakes['B'])} segments)")
        print(f"🍎 FOODS CONSUMED:")
        print(f"   Snake A: {foods_eaten_a} foods")
        print(f"   Snake B: {foods_eaten_b} foods")
        print(f"🎮 EPISODE STATISTICS:")
        print(f"   Total Frames: {len(episode.frames)}")
        print(f"   Championship Performance: {episode.final_score}-point episode")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading replay: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = play_replay_interactive()
    if success:
        print("\n🎉 Thanks for watching the championship replay!")
    else:
        print("\n❌ Replay failed to load")
        exit(1)