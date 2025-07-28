import time
from typing import Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class ConversationStateTracker:
    """Tracks conversation state to determine dynamic interruption behavior."""
    
    def __init__(self, continuity_threshold: float = 8.0):
        self.last_user_speech_end_time: Optional[float] = None
        self.last_agent_speech_start_time: Optional[float] = None
        self.continuity_threshold = continuity_threshold
        self.enabled = True
        # Track recent continuation collisions for adaptive endpointing
        self.collision_memory = deque()  # Stores timestamps of collisions
    
    def update_user_speech_ended(self, timestamp: Optional[float] = None):
        """Called when user finishes speaking."""
        self.last_user_speech_end_time = timestamp or time.time()
    
    def update_agent_speech_started(self, timestamp: Optional[float] = None):
        """Called when agent starts speaking."""
        self.last_agent_speech_start_time = timestamp or time.time()
    
    def get_dynamic_min_interruption_words(self, current_time: Optional[float] = None) -> int:
        """
        Returns dynamic min_interruption_words based on conversation context.
        
        Returns:
            0: During conversation flow (immediate interruption on VAD)
            1: Fresh start or after long pause (word-based interruption)
        """
        if not self.enabled:
            return 1  # Fall back to original behavior
        
        if not self.last_user_speech_end_time:
            return 1  # Fresh start
        
        current_time = current_time or time.time()
        time_since_user_speech = current_time - self.last_user_speech_end_time
        
        # Account for agent response latency
        adjusted_threshold = self.continuity_threshold
        if self.last_agent_speech_start_time and self.last_user_speech_end_time:
            agent_response_time = self.last_agent_speech_start_time - self.last_user_speech_end_time
            # Add extra buffer for longer response times
            adjusted_threshold += max(0, agent_response_time - 2.0)
        
        # Return 0 for conversation flow (immediate interruption), 1 for fresh starts
        return 0 if time_since_user_speech <= adjusted_threshold else 1
    
    def is_in_conversation_flow(self, current_time: Optional[float] = None) -> bool:
        """Check if we're currently in conversation flow."""
        return self.get_dynamic_min_interruption_words(current_time) == 0
    
    def record_continuation_collision(self):
        """Record when user interrupts agent because they needed more time."""
        now = time.time()
        self.collision_memory.append(now)
        # Keep only collisions from last 15 seconds
        cutoff = now - 15.0
        self.collision_memory = deque(t for t in self.collision_memory if t > cutoff)
    
    def get_endpointing_multiplier(self) -> float:
        """Get multiplier for endpointing delay based on recent collisions."""
        if not self.enabled:
            return 1.0
            
        # Count recent collisions (last 15 seconds)
        now = time.time()
        cutoff = now - 15.0
        recent_collisions = sum(1 for t in self.collision_memory if t > cutoff)
        
        # More collisions = user needs more time
        if recent_collisions >= 2:
            return 3.0  # Triple the delay
        elif recent_collisions >= 1:
            return 2.0  # Double the delay
        return 1.0  # No recent collisions, normal timing


class DynamicInterruptionManager:
    """Manages dynamic interruption logic for voice sessions."""
    
    def __init__(self, session_options):
        self.session_options = session_options
        self.conversation_state = ConversationStateTracker(
            continuity_threshold=getattr(session_options, 'conversation_continuity_threshold', 8.0)
        )
        self.conversation_state.enabled = getattr(session_options, 'enable_dynamic_interruption', True)
        
        # Store original min_interruption_words
        self.original_min_interruption_words = session_options.min_interruption_words
    
    def get_current_min_interruption_words(self) -> int:
        """Get the current min_interruption_words value based on conversation context."""
        if not self.conversation_state.enabled:
            return self.original_min_interruption_words
        
        dynamic_value = self.conversation_state.get_dynamic_min_interruption_words()
        
        # If original was 0, respect that (don't increase it)
        if self.original_min_interruption_words == 0:
            return 0
        
        return dynamic_value
    
    def on_user_speech_ended(self):
        """Called when user finishes speaking."""
        self.conversation_state.update_user_speech_ended()
    
    def on_agent_speech_started(self):
        """Called when agent starts speaking."""
        self.conversation_state.update_agent_speech_started()
    
    def record_continuation_collision(self):
        """Record when user interrupts agent because they needed more time."""
        self.conversation_state.record_continuation_collision()
    
    def get_endpointing_multiplier(self) -> float:
        """Get multiplier for endpointing delay based on recent collisions."""
        return self.conversation_state.get_endpointing_multiplier()