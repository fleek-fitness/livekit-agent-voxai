from __future__ import annotations

import time
from dataclasses import dataclass

# Typed only for clarity; avoid importing to reduce risk of circular imports at import time
try:  # pragma: no cover - type hint convenience
    from .agent_session import VoiceOptions  # type: ignore
except Exception:  # pragma: no cover
    VoiceOptions = object  # type: ignore


@dataclass
class ConversationStateTracker:
    last_user_speech_end_time: float | None = None
    continuation_collisions: int = 0


class DynamicInterruptionManager:
    def __init__(self, opts: VoiceOptions) -> None:
        self._opts = opts
        self.conversation_state = ConversationStateTracker()

    # ---- Interruption dynamics ----
    def on_agent_speech_started(self) -> None:
        # Today no-op; reserved for future heuristics (e.g., decay collisions on speak)
        pass

    def on_user_speech_ended(self) -> None:
        self.conversation_state.last_user_speech_end_time = time.time()

    def get_current_min_interruption_words(self) -> int:
        """Compute an adaptive min_interruption_words threshold.

        Rationale:
        - If the user's speech resumes within a short continuity window, allow instant interruption
          by returning 0 words requirement.
        - Otherwise, require at least 1 word (or the configured static minimum if larger).
        """
        # Feature gate
        if not getattr(self._opts, "enable_dynamic_interruption", False):
            return getattr(self._opts, "min_interruption_words", 0)

        last = self.conversation_state.last_user_speech_end_time
        base_min = int(getattr(self._opts, "min_interruption_words", 0) or 0)
        threshold = float(getattr(self._opts, "conversation_continuity_threshold", 8.0) or 8.0)

        if last is None:
            return base_min

        if (time.time() - last) <= threshold:
            return 0

        return max(1, base_min)

    # ---- Endpointing dynamics ----
    @property
    def adaptive_endpointing_enabled(self) -> bool:
        return bool(getattr(self._opts, "enable_adaptive_endpointing", False))

    def record_continuation_collision(self) -> None:
        self.conversation_state.continuation_collisions += 1

    def get_endpointing_multiplier(self) -> float:
        """Return a backoff multiplier based on recent collisions.

        A simple capped linear backoff: 1.0 + 0.5 per collision, up to +1.5x.
        """
        if not self.adaptive_endpointing_enabled:
            return 1.0

        collisions = int(self.conversation_state.continuation_collisions)
        if collisions <= 0:
            return 1.0

        return 1.0 + min(collisions, 3) * 0.5

    def reset_collisions(self) -> None:
        self.conversation_state.continuation_collisions = 0


