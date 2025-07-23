from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from .agent_session import VoiceOptions


@dataclass
class ConversationStateTracker:
    """Tracks conversation state to determine dynamic interruption behavior."""
    
    def __init__(self, continuity_threshold: float = 8.0):
        self.last_user_speech_end_time: Optional[float] = None
        self.last_agent_speech_start_time: Optional[float] = None
        self.continuity_threshold = continuity_threshold
        self.enabled = True
    
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


class DynamicInterruptionManager:
    """Manages dynamic interruption logic for voice sessions."""
    
    def __init__(self, session_options: VoiceOptions) -> None:
        self.session_options = session_options
        self.conversation_state = ConversationStateTracker(
            continuity_threshold=getattr(session_options, 'conversation_continuity_threshold', 8.0)
        )
        self.conversation_state.enabled = getattr(session_options, 'enable_dynamic_interruption', True)
        
        # Store original min_interruption_words
        self.original_min_interruption_words = session_options.min_interruption_words
        
        # Interruption history tracking for context-aware endpointing
        self._interruption_history: list[float] = []
    
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
    
    def on_interruption(self) -> None:
        """Called when an interruption/overlap is detected"""
        current_time = time.time()
        self._interruption_history.append(current_time)
    
    def get_endpointing_delay_multiplier(self) -> float:
        """Returns multiplier for endpointing delays based on recent interruptions"""
        if not self._interruption_history:
            return 1.0  # Normal delays when no interruptions
        
        # Be more patient for 10 seconds after most recent interruption  
        most_recent = self._interruption_history[-1]
        time_since_interruption = time.time() - most_recent
        
        if time_since_interruption < 10.0:
            return 2.0  # 2x patience after recent interruption
        else:
            return 1.0  # Return to normal responsiveness
    
