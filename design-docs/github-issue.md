# [Enhancement] Dynamic Interruption System for Natural Voice AI Conversations

## 🎯 Problem Statement

The current interruption system in LiveKit voice agents creates unnatural "ping-pong" conversations where users and AI agents frequently talk over each other, resulting in poor user experience and awkward conversational flow.

### 📋 Current Behavior Analysis

The existing implementation uses a fixed `min_interruption_words=1` configuration to prevent background noise from triggering false interruptions. While this successfully filters out ambient sounds (dogs barking, subway noise, etc.), it creates significant UX problems:

1. **Latency Impact**: The STT→LLM→TTS pipeline introduces 1-2 seconds of response latency
2. **Conversation Collisions**: Users naturally pause mid-sentence to think (3-5 seconds), during which the agent begins responding
3. **Interruption Delay**: When users resume speaking, they must complete at least one word before the agent stops, creating overlapping speech
4. **Repeated Collisions**: This creates a "walking past someone in a hallway" effect where both parties keep starting and stopping

### 🔍 Root Cause Analysis

The fundamental issue is treating all speech contexts identically. The current rule-based system cannot distinguish between:
- **Fresh conversation starts** (where noise filtering is crucial)
- **Ongoing conversation flow** (where immediate interruption is natural)

### 🎬 Specific Problem Scenario

**Current Behavior Flow:**
1. User: "I want to..." (pauses 3 seconds to think)
2. Agent detects silence → starts STT→LLM→TTS pipeline
3. After 1-2 seconds latency, Agent starts speaking: "What would you like..."
4. User resumes: "...buy some apples"
5. VAD detects user speaking, but Agent waits for STT to produce **1 full word**
6. **Result**: Both speaking simultaneously until STT transcribes the first word
7. This creates the "와리가리" (ping-pong) problem

## 🛠️ Technical Analysis

### Current Implementation

Based on codebase analysis, the interruption logic is implemented in `agent_activity.py` with two key methods:

1. **`on_vad_inference_done()`** (lines 869-918): VAD-based interruption checking
2. **`on_end_of_turn()`** (lines 939-985): End-of-turn interruption handling

Both methods access `self._session.options.min_interruption_words` and use `split_words(text, split_character=True)` for word counting.

**Key Code Sections:**

```python
# In on_vad_inference_done()
if (
    self.stt is not None
    and self._session.options.min_interruption_words > 0
    and self._audio_recognition is not None
):
    text = self._audio_recognition.current_transcript
    if (
        len(split_words(text, split_character=True))
        < self._session.options.min_interruption_words
    ):
        return
```

## 💡 Proposed Solution: Dynamic Interruption System

### Architecture Overview

For a forked library, the optimal approach is to create a separate file containing all dynamic interruption logic, with minimal integration points in the existing codebase.

### Core Solution Logic

**Key Insight**: The problem isn't the agent starting to speak during user pauses (that's expected), but rather the delay in stopping when the user resumes speaking. By making `min_interruption_words = 0` during conversation flow, the agent stops immediately on VAD detection without waiting for STT words.

### Implementation Strategy

#### 1. Create New File: `dynamic_interruption.py`

```python
import time
from typing import Optional
from dataclasses import dataclass

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
```

#### 2. Integration Points in Existing Code

##### A. Modify `agent_activity.py`

**Add dynamic interruption manager initialization:**

```python
# In AgentActivity.__init__()
def __init__(self, agent: Agent, sess: AgentSession) -> None:
    # ... existing initialization ...
    
    # Import at the top of file
    from .dynamic_interruption import DynamicInterruptionManager
    
    # Add dynamic interruption manager
    self._dynamic_interruption = DynamicInterruptionManager(sess.options)
```

##### B. Hook into Speech Events

```python
# In on_end_of_speech() - Track when user finishes speaking
def on_end_of_speech(self, ev: vad.VADEvent) -> None:
    self._session._update_user_state("listening")
    # Add dynamic interruption hook - track when user stops speaking
    self._dynamic_interruption.on_user_speech_ended()

# Hook into agent speech start events
def _update_agent_state(self, state: AgentState) -> None:
    # ... existing logic ...
    if state == "speaking":
        self._dynamic_interruption.on_agent_speech_started()
```

##### C. Modify Interruption Logic

```python
# In on_vad_inference_done()
def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
    # ... existing checks ...
    
    if (
        self.stt is not None
        and self._dynamic_interruption.get_current_min_interruption_words() > 0  # Use dynamic value
        and self._audio_recognition is not None
    ):
        text = self._audio_recognition.current_transcript
        
        if (
            len(split_words(text, split_character=True))
            < self._dynamic_interruption.get_current_min_interruption_words()  # Use dynamic value
        ):
            return
    
    # ... rest of existing logic ...

# In on_end_of_turn()
def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
    # ... existing checks ...
    
    if (
        self.stt is not None
        and self._turn_detection_mode != "manual"
        and self._current_speech is not None
        and self._current_speech.allow_interruptions
        and not self._current_speech.interrupted
        and self._dynamic_interruption.get_current_min_interruption_words() > 0  # Use dynamic value
        and len(split_words(info.new_transcript, split_character=True))
        < self._dynamic_interruption.get_current_min_interruption_words()  # Use dynamic value
    ):
        return False
    
    # ... rest of existing logic ...
```

##### D. Configuration Options in `agent_session.py`

```python
# Add to VoiceOptions dataclass
@dataclass
class VoiceOptions:
    # ... existing options ...
    
    # Dynamic interruption settings
    enable_dynamic_interruption: bool = True
    conversation_continuity_threshold: float = 8.0  # seconds
```

##### E. Add to AgentSession constructor

```python
# In AgentSession.__init__()
def __init__(
    self,
    # ... existing parameters ...
    enable_dynamic_interruption: bool = True,
    conversation_continuity_threshold: float = 8.0,
    # ... existing parameters ...
) -> None:
    # ... existing initialization ...
    
    self._opts = VoiceOptions(
        # ... existing options ...
        enable_dynamic_interruption=enable_dynamic_interruption,
        conversation_continuity_threshold=conversation_continuity_threshold,
    )
```

## ✅ Solution Verification

### How This Fixes the "와리가리" Problem

**Before (Current Behavior):**
1. User: "I want to..." (pauses 3 seconds to think)
2. Agent starts speaking: "What would you like..."
3. User resumes: "...buy some apples"
4. VAD detects user speaking, but Agent waits for STT to produce **1 full word**
5. **Result**: Both speaking simultaneously until STT transcribes the first word

**After (With Dynamic Interruption):**
1. User: "I want to..." (pauses 3 seconds) 
2. System records: `last_user_speech_end_time = when_user_stopped_saying_"want"`
3. Agent starts speaking: "What would you like..." (expected behavior)
4. User resumes: "...buy some apples"
5. System calculates: `time_since_user_speech = 3 seconds < 8 seconds`
6. **Therefore: `min_interruption_words = 0`** (conversation flow mode)
7. Agent stops **immediately** when VAD detects user speaking (no need to wait for STT words)

**🎯 Key Improvement**: Eliminates the STT word-waiting delay during conversation flow, directly solving the overlapping speech problem.

## 🚀 Implementation Benefits

### 1. Forked Library Advantages
- **Isolated Code**: All new logic in separate file
- **Minimal Changes**: Only essential integration points modified
- **Easy Maintenance**: Clear separation between original and custom code
- **Future-Proof**: Easy to merge upstream changes

### 2. Backward Compatibility
- **Configurable**: Can be enabled/disabled via configuration
- **Graceful Degradation**: Falls back to original behavior if disabled
- **No Breaking Changes**: Existing API remains unchanged

### 3. Technical Benefits
- **Natural Conversations**: Eliminates "ping-pong" effect by removing STT wait time
- **Context Awareness**: Distinguishes conversation flow from fresh starts
- **Adaptive Behavior**: Accounts for agent response latency
- **Robust Design**: Handles edge cases and network delays

## 🧪 Testing Strategy

### Unit Tests
- [ ] Conversation state transitions
- [ ] Dynamic threshold calculations
- [ ] Edge case handling (rapid speech, network delays)
- [ ] Backward compatibility verification

### Integration Tests
- [ ] End-to-end conversation flow scenarios
- [ ] Performance impact measurement
- [ ] Different conversation patterns (short pauses, long thinking pauses)

### User Experience Testing
- [ ] A/B testing against current implementation
- [ ] Conversation naturalness metrics
- [ ] Interruption appropriateness scoring

## 📅 Implementation Timeline

### Phase 1: Core Implementation (Week 1-2)
- [ ] Create `dynamic_interruption.py` with conversation state tracker
- [ ] Add configuration options to `VoiceOptions`
- [ ] Implement minimal integration points in `agent_activity.py`
- [ ] Add comprehensive unit tests

### Phase 2: Integration & Testing (Week 3)
- [ ] Integration testing with existing codebase
- [ ] Performance optimization
- [ ] Edge case handling
- [ ] Documentation updates

### Phase 3: Validation & Refinement (Week 4)
- [ ] User experience testing
- [ ] Conversation flow analysis
- [ ] Fine-tuning of thresholds
- [ ] Final optimization

## 📊 Success Metrics

### Quantitative Metrics
- **Conversation Collisions**: Target 80% reduction in overlapping speech
- **Response Latency**: Maintain current performance
- **Interruption Accuracy**: 95% appropriate interruptions
- **Code Maintainability**: Minimal changes to original codebase

### Qualitative Metrics
- **User Satisfaction**: Improved conversation naturalness
- **Developer Experience**: Easy to configure and maintain
- **System Reliability**: Robust performance across scenarios

## 🔧 Configuration Options

### New Configuration Parameters

```python
# Enable/disable dynamic interruption
enable_dynamic_interruption: bool = True

# Time threshold for conversation continuity (seconds)
conversation_continuity_threshold: float = 8.0
```

### Usage Example

```python
session = AgentSession(
    # ... existing parameters ...
    enable_dynamic_interruption=True,
    conversation_continuity_threshold=8.0,
    # ... other parameters ...
)
```

## 🔄 Backward Compatibility

- **Default Behavior**: Feature enabled by default but can be disabled
- **Graceful Fallback**: Falls back to original behavior when disabled
- **No Breaking Changes**: All existing APIs remain unchanged
- **Migration Path**: No migration required for existing implementations

## 📝 Additional Considerations

### Edge Cases Handled
- **Network Delays**: Accounts for variable response times
- **Rapid Speech**: Handles quick back-and-forth conversations
- **Long Pauses**: Reverts to original behavior after threshold
- **Multiple Users**: Isolated per session

### Performance Impact
- **Minimal Overhead**: Simple time-based calculations
- **Memory Efficient**: Lightweight state tracking
- **CPU Impact**: Negligible additional processing

## 🎯 Expected Outcomes

1. **Eliminates "와리가리" Problem**: Agent stops immediately on VAD detection during conversation flow
2. **Maintainable Code**: Clear separation of concerns
3. **Backward Compatibility**: No breaking changes
4. **Future-Proof Design**: Easy to enhance and maintain

The implementation directly addresses the core issue: the delay between user resuming speech and agent stopping, which creates unnatural overlapping conversations. By making `min_interruption_words = 0` during conversation flow, we eliminate this delay and create natural, human-like conversational patterns.

## 🏷️ Labels

- `enhancement`
- `voice-ai`
- `conversation-flow`
- `user-experience`
- `interruption-handling`

## 🔗 Related Issues

- None currently identified

---

**Priority**: High
**Complexity**: Medium
**Impact**: High User Experience Improvement