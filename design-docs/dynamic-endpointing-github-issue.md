# Implement Context-Aware Dynamic Endpointing to Resolve Conversation Deadlocks

## Problem Statement

### Current Issue: "Wari-gari" Conversation Deadlocks

Voice AI conversations frequently experience overlapping speech patterns that create cascading interruption cycles, similar to two people trying to pass each other in a narrow hallway. This manifests as:

1. **Initial Trigger**: AI response latency (~2 seconds) causes users to speak before AI responds
2. **Natural Speech Misinterpretation**: Users naturally pause during speech (e.g., phone numbers: "010... pause... 9507... pause... 5331"), but AI interprets these pauses as end-of-turn
3. **Cascading Deadlock**: Once overlapping begins, both parties continuously interrupt each other
4. **User Frustration**: Users become increasingly assertive, exacerbating the problem

### Current System Limitations

The existing system has sophisticated infrastructure but lacks conversation context awareness:

- **Static Endpointing**: Fixed delays (min=0.5s, max=1.5s) don't adapt to conversation state
- **Isolated Decision Making**: `AudioRecognition._bounce_eou_task()` makes endpointing decisions without conversation context
- **Existing Dynamic System**: `DynamicInterruptionManager` tracks conversation state but only handles user→agent interruptions, not agent→user endpointing
- **Turn Detector Reliability**: ML model exists but has accuracy issues, forcing conservative configurations

## Root Cause Analysis

The core issue is **architectural separation**: `AudioRecognition` (responsible for endpointing) and `DynamicInterruptionManager` (which tracks conversation context) don't communicate. This results in context-blind endpointing decisions.

**Key Files Involved:**
- `livekit/agents/voice/audio_recognition.py:314-356` - `_bounce_eou_task()` method
- `livekit/agents/voice/dynamic_interruption.py` - Conversation state tracking
- `livekit/agents/voice/agent_activity.py:456-467` - Component integration

## Proposed Solution: Simple Adaptive Endpointing

### Core Concept

**Single Parameter Solution**: Add one boolean flag that makes endpointing delays conversation-aware using existing infrastructure.

### Minimal Architecture Change

#### 1. Extend DynamicInterruptionManager (10 lines of code)

```python
class DynamicInterruptionManager:
    def __init__(self):
        self._interruption_history: list[float] = []  # Track all interruption timestamps
        self._history_window = 60.0  # Keep 60 seconds of history
        
    def on_interruption_occurred(self):
        """Called when an interruption/overlap is detected"""
        current_time = time.time()
        self._interruption_history.append(current_time)
        
        # Clean old history beyond window
        cutoff_time = current_time - self._history_window
        self._interruption_history = [t for t in self._interruption_history if t > cutoff_time]
    
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
    
    def get_interruption_frequency(self, window_seconds: float = 30.0) -> float:
        """Get interruptions per minute in the specified window"""
        if not self._interruption_history:
            return 0.0
        
        cutoff_time = time.time() - window_seconds
        recent_interruptions = [t for t in self._interruption_history if t > cutoff_time]
        return len(recent_interruptions) * (60.0 / window_seconds)  # per minute
```

#### 2. Create SpeechInterruptedEvent (follows existing event patterns)

```python
# In events.py - add new event type following existing patterns

class SpeechInterruptedEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["speech_interrupted"] = "speech_interrupted"
    speech_handle: SpeechHandle
    timestamp: float
```

#### 3. Emit Event from SpeechHandle (leverage existing event system)

```python
# In speech_handle.py - add session reference and emit event

class SpeechHandle:
    def __init__(self, *, speech_id: str, allow_interruptions: bool, 
                 step_index: int, parent: SpeechHandle | None,
                 session: AgentSession | None = None):  # NEW
        # existing init...
        self._session = session
    
    @staticmethod
    def create(allow_interruptions: bool = True, step_index: int = 0,
               parent: SpeechHandle | None = None, 
               session: AgentSession | None = None):  # NEW
        return SpeechHandle(
            speech_id=utils.shortuuid("speech_"),
            allow_interruptions=allow_interruptions,
            step_index=step_index,
            parent=parent,
            session=session  # NEW
        )

    def interrupt(self) -> SpeechHandle:
        """Interrupt the current speech generation."""
        if not self._allow_interruptions:
            raise RuntimeError("This generation handle does not allow interruptions")

        if self.done():
            return self

        with contextlib.suppress(asyncio.InvalidStateError):
            self._interrupt_fut.set_result(None)
        
        # NEW: Emit interruption event through existing event system
        if self._session:
            self._session.emit("speech_interrupted", 
                SpeechInterruptedEvent(speech_handle=self, timestamp=time.time())
            )
        
        return self
```

#### 4. Listen to Events in DynamicInterruptionManager (clean observer pattern)

```python
# In dynamic_interruption.py - subscribe to interruption events

class DynamicInterruptionManager:
    def __init__(self, session: AgentSession):
        self._session = session
        self._interruption_history: list[float] = []
        self._history_window = 60.0
        
        # Subscribe to interruption events using existing event system
        session.on("speech_interrupted", self._on_speech_interrupted)
    
    def _on_speech_interrupted(self, event: SpeechInterruptedEvent):
        """Event handler for speech interruptions"""
        self.on_interruption_occurred()
    
    # rest of implementation unchanged...
```

#### 5. Modify AudioRecognition Endpointing Logic (4 lines of code)

```python
async def _bounce_eou_task(self, last_speaking_time: float):
    # Existing ML model logic unchanged
    base_delay = self._min_endpointing_delay
    if turn_detector and end_of_turn_probability < unlikely_threshold:
        base_delay = self._max_endpointing_delay
    
    # NEW: Apply interruption-aware multiplier 
    if self._dynamic_interruption:
        multiplier = self._dynamic_interruption.get_endpointing_delay_multiplier()
        endpointing_delay = base_delay * multiplier
        # Apply safety bounds
        endpointing_delay = min(endpointing_delay, self._max_endpointing_delay * 2.0)
    else:
        endpointing_delay = base_delay
    
    # Continue with existing delay logic...
```

### Two Simple States

1. **NO_RECENT_INTERRUPTIONS**: Multiplier = 1.0x (normal responsive turn-taking)
2. **RECENT_INTERRUPTIONS_DETECTED**: Multiplier = 2.0x (step back behavior, be more patient)

### Single Configuration Option

```python
@dataclass  
class VoiceOptions:
    # Existing parameters unchanged...
    
    # NEW: Single parameter
    enable_conversation_aware_endpointing: bool = True
```

## Implementation Plan

### Single Phase: Simple Solution (3 days)
- [ ] Create `SpeechInterruptedEvent` in events.py (2 minutes)
- [ ] Add session reference to `SpeechHandle` constructor and emit event (5 minutes)
- [ ] Add event listener to `DynamicInterruptionManager` (5 minutes)
- [ ] Pass session to `SpeechHandle.create()` calls in `AgentActivity` (2 minutes)
- [ ] Modify `AudioRecognition._bounce_eou_task()` logic (5 minutes) 
- [ ] Pass `DynamicInterruptionManager` to `AudioRecognition` constructor (5 minutes)
- [ ] Add single boolean configuration parameter (5 minutes)
- [ ] Test "wari-gari" overlapping scenarios (2 days)
- [ ] Deploy with feature flag (1 day)

## Success Metrics

**Primary KPIs:**
1. **Reduction in overlapping speech events** (target: 60% reduction)
2. **Improved turn completion rates** (fewer abandoned conversations)
3. **Reduced user frustration indicators** (fewer repeated interruptions)

**Secondary Metrics:**
4. **Maintained responsiveness** (average response latency unchanged)
5. **Natural conversation flow** (smoother turn transitions)
6. **Configuration effectiveness** (boolean flag usage analysis)

## Technical Benefits

### Elegance Through Existing Infrastructure
- **Event-Driven Architecture**: Uses existing `session.emit()` and event listener patterns
- **Zero Coupling**: No contextvar hacks - clean observer pattern with events
- **Follows Established Patterns**: `SpeechInterruptedEvent` matches existing event conventions
- **Single Source of Truth**: All interruptions flow through `SpeechHandle.interrupt()`
- **Testable & Observable**: Events can be easily mocked and monitored
- **Minimal Performance Impact**: Lightweight event emission, no additional ML inference

### Robustness Considerations
- **Language Support**: Works with existing turn detector language support
- **Network Resilience**: Uses existing min/max delay bounds
- **Clean Dependencies**: DynamicInterruptionManager subscribes to events, no tight coupling
- **Safe Defaults**: Uses proven max_endpointing_delay values
- **Event System Benefits**: Automatic error handling, async-safe, established patterns

## Edge Cases Handled

1. **Natural Speech Patterns**: Phone numbers, addresses handled automatically when no recent interruptions
2. **Cascading Interruptions**: After first interruption, system becomes more patient for 10 seconds  
3. **False Interruption Recovery**: 10-second cooldown allows system to return to normal responsiveness
4. **System Performance**: Simple timestamp check, zero overhead

## Files to Modify

1. **`livekit/agents/voice/events.py`**
   - Add `SpeechInterruptedEvent` class following existing patterns (4 lines)

2. **`livekit/agents/voice/speech_handle.py`**
   - Add session parameter to `__init__()` and `create()` methods (3 lines)
   - Emit `speech_interrupted` event in `interrupt()` method (3 lines)

3. **`livekit/agents/voice/dynamic_interruption.py`**
   - Add interruption history tracking to `__init__()` (2 lines)
   - Add event listener in `__init__()` (1 line)
   - Add `_on_speech_interrupted()` event handler (2 lines)
   - Add `on_interruption_occurred()` method with history management (5 lines)
   - Add `get_endpointing_delay_multiplier()` method (8 lines)  
   - Add `get_interruption_frequency()` method for analytics (6 lines)

4. **`livekit/agents/voice/agent_activity.py`**
   - Pass session to `SpeechHandle.create()` calls (2 lines)
   - Pass `DynamicInterruptionManager` to `AudioRecognition` (1 line)

5. **`livekit/agents/voice/audio_recognition.py`**
   - Modify `_bounce_eou_task()` logic with multiplier approach (6 lines)
   - Add `DynamicInterruptionManager` parameter (1 line)

6. **`livekit/agents/voice/agent_session.py`**
   - Add single boolean parameter to `VoiceOptions` (1 line)

## Testing Strategy

- **"Wari-gari" Deadlock Test**: Simulate overlapping speech scenarios
- **Recovery Test**: Verify system returns to normal after 10 seconds
- **Interruption History Test**: Confirm all interruptions are tracked and cleaned up
- **Frequency Analytics Test**: Validate interruption frequency calculations
- **Feature Flag Test**: Enable/disable behavior validation

## Risk Mitigation

- **Feature Flag**: Easy rollback capability  
- **Single Boolean**: Minimal configuration complexity
- **Existing Infrastructure**: Uses proven delay values
- **Zero New Parameters**: No tuning required

---

This solution addresses the core "wari-gari" deadlock problem by implementing human-like "step back" behavior. When interruptions occur, the AI becomes more patient for 10 seconds, exactly like humans do in narrow hallways. The implementation leverages existing infrastructure while directly targeting the cascading interruption cycles described in the transcript.