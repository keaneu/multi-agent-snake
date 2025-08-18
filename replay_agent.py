"""
Replay Agent - Records high-score game episodes and replays them
"""

import json
import time
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from multi_agent_framework import BaseAgent, MessageType, Message

@dataclass
class ReplayFrame:
    """A single frame of a replay."""
    snakes: Dict[str, List[tuple]]
    food: List[tuple]
    obstacles: List[tuple]
    scores: Dict[str, int]
    game_phase: str

@dataclass
class ReplayEpisode:
    """A full episode for replay."""
    episode_id: str
    final_score: int
    frames: List[ReplayFrame] = field(default_factory=list)

class ReplayAgent(BaseAgent):
    """
    Records game sessions and can replay high-score episodes.
    """
    def __init__(self, high_score_threshold: int = 100):
        super().__init__("replay_agent")
        self.high_score_threshold = high_score_threshold
        self.record_mode = True
        self.replay_mode = False
        self.current_episode_frames: List[ReplayFrame] = []
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15  # Record at 15 FPS

        # State for recording
        self.snakes = {}
        self.food = []
        self.obstacles = []
        self.scores = {}
        self.game_phase = "initializing"

        # Ensure replay directory exists
        if not os.path.exists("replays"):
            os.makedirs("replays")

    def initialize(self):
        """Initialize the replay agent."""
        print("🎥 Replay Agent initializing...")
        self.subscribe_to_messages([
            MessageType.GAME_STATE,
            MessageType.ENVIRONMENT_UPDATE,
            MessageType.GAME_OVER,
            MessageType.RESET_GAME,
            MessageType.START_REPLAY,
        ])
        self.send_message(MessageType.AGENT_READY, "game_engine", {"agent_id": self.agent_id})

    def update(self):
        """Update loop for the replay agent."""
        current_time = time.time()
        if self.record_mode and not self.replay_mode:
            if current_time - self.last_frame_time >= self.frame_interval:
                self.record_frame()
                self.last_frame_time = current_time

        if self.replay_mode:
            # Replay logic will be triggered by a message
            pass

    def process_message(self, message: Message):
        """Process incoming messages."""
        if message.type == MessageType.GAME_STATE:
            self.handle_game_state(message.data)
        elif message.type == MessageType.ENVIRONMENT_UPDATE:
            self.handle_environment_update(message.data)
        elif message.type == MessageType.GAME_OVER:
            self.handle_game_over(message.data)
        elif message.type == MessageType.RESET_GAME:
            self.handle_game_reset(message.data)
        elif message.type == MessageType.START_REPLAY:
            self.start_replay()

    def handle_game_state(self, data: dict):
        """Handle game state updates for recording."""
        update_type = data.get("type")
        if update_type == "snake_update":
            snake_id = data.get("snake_id")
            if snake_id:
                self.snakes[snake_id] = data.get("segments", [])
        elif update_type == "update":
            self.game_phase = data.get("phase", self.game_phase)
            players = data.get("players", {})
            for pid, pdata in players.items():
                self.scores[pid] = pdata.get("score", 0)

    def handle_environment_update(self, data: dict):
        """Handle environment updates for recording."""
        self.food = data.get("foods", [])
        self.obstacles = data.get("obstacles", [])

    def record_frame(self):
        """Record the current state as a single frame."""
        if self.game_phase != "running":
            return

        frame = ReplayFrame(
            snakes=self.snakes.copy(),
            food=self.food.copy(),
            obstacles=self.obstacles.copy(),
            scores=self.scores.copy(),
            game_phase=self.game_phase
        )
        self.current_episode_frames.append(frame)

    def handle_game_over(self, data: dict):
        """Handle game over to save high-score episodes."""
        final_scores = data.get("final_scores", {})
        total_score = sum(final_scores.values()) if final_scores else 0
        max_individual_score = max(final_scores.values()) if final_scores else 0

        # Save if either total score OR any individual score exceeds threshold
        if total_score > self.high_score_threshold or max_individual_score > self.high_score_threshold:
            score_to_save = max(total_score, max_individual_score)
            print(f"🎥 HIGH SCORE DETECTED! Saving episode with score {score_to_save}")
            self.save_episode(score_to_save)

    def save_episode(self, final_score: int):
        """Save the recorded episode to a file."""
        if not self.current_episode_frames:
            print("⚠️  No frames recorded for episode")
            return

        episode_id = f"episode_{int(time.time())}_score_{final_score}"
        episode = ReplayEpisode(
            episode_id=episode_id,
            final_score=final_score,
            frames=[frame.__dict__ for frame in self.current_episode_frames]
        )

        filename = f"replays/{episode_id}.json"
        with open(filename, "w") as f:
            json.dump(episode.__dict__, f, indent=2)

        print(f"🎥 Saved high-score episode {episode_id} with score {final_score} to {filename}")
        print(f"📊 Episode contains {len(self.current_episode_frames)} frames")

    def handle_game_reset(self, data: dict):
        """Handle game reset to start a new recording."""
        self.current_episode_frames = []
        self.snakes = {}
        self.scores = {}
        print("🎥 Cleared episode recording for new game.")

    def start_replay(self):
        """Finds the latest replay and starts replaying it in a new thread."""
        if self.replay_mode:
            print("🎥 Replay already in progress.")
            return

        print("🎬 Searching for replay files...")
        replay_files = [f for f in os.listdir("replays") if f.endswith(".json")]
        if not replay_files:
            print("❌ No replay files found.")
            return

        latest_replay_file = max(replay_files, key=lambda f: os.path.getmtime(os.path.join("replays", f)))
        filepath = os.path.join("replays", latest_replay_file)

        print(f"🎬 Loading replay from {filepath}...")
        try:
            with open(filepath, "r") as f:
                replay_data = json.load(f)

            episode = ReplayEpisode(
                episode_id=replay_data["episode_id"],
                final_score=replay_data["final_score"],
                frames=[ReplayFrame(**frame) for frame in replay_data["frames"]]
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to load or parse replay file: {e}")
            return

        self.replay_mode = True
        self.record_mode = False

        # Pause the game
        self.broadcast_message(MessageType.PAUSE_GAME, {"paused": True})

        # Start replay in a new thread
        replay_thread = threading.Thread(target=self.replay_loop, args=(episode,), daemon=True)
        replay_thread.start()

    def replay_loop(self, episode: ReplayEpisode):
        """The main loop for replaying an episode."""
        print(f"▶️  Playing episode {episode.episode_id} with {len(episode.frames)} frames...")

        for frame in episode.frames:
            if not self.replay_mode:
                print("⏹️ Replay stopped prematurely.")
                break

            # Send frame data to visualization agent
            self.send_message(MessageType.REPLAY_FRAME, "visualization", frame.__dict__)
            time.sleep(self.frame_interval)

        print("🏁 Replay finished.")
        self.stop_replay()

    def stop_replay(self):
        """Stops the current replay and resumes the game."""
        print("🎬 Stopping replay and resuming game...")
        self.replay_mode = False
        self.record_mode = True
        # Un-pause the game
        self.broadcast_message(MessageType.PAUSE_GAME, {"paused": False})
