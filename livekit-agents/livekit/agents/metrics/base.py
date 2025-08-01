from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class LLMMetrics(BaseModel):
    type: Literal["llm_metrics"] = "llm_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    ttft: float
    cancelled: bool
    completion_tokens: int
    prompt_tokens: int
    prompt_cached_tokens: int
    total_tokens: int
    tokens_per_second: float
    speech_id: str | None = None


class STTMetrics(BaseModel):
    type: Literal["stt_metrics"] = "stt_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    """The request duration in seconds, 0.0 if the STT is streaming."""
    audio_duration: float
    """The duration of the pushed audio in seconds."""
    streamed: bool
    """Whether the STT is streaming (e.g using websocket)."""


class TTSMetrics(BaseModel):
    type: Literal["tts_metrics"] = "tts_metrics"
    label: str
    request_id: str
    timestamp: float
    ttfb: float
    duration: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    streamed: bool
    speech_id: str | None = None


class VADMetrics(BaseModel):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int


class EOUMetrics(BaseModel):
    type: Literal["eou_metrics"] = "eou_metrics"
    timestamp: float
    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn."""

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech."""

    on_user_turn_completed_delay: float
    """Time taken to invoke the user's `Agent.on_user_turn_completed` callback."""

    speech_id: str | None = None


class RealtimeModelMetrics(BaseModel):
    class CachedTokenDetails(BaseModel):
        audio_tokens: int
        text_tokens: int
        image_tokens: int

    class InputTokenDetails(BaseModel):
        audio_tokens: int
        text_tokens: int
        image_tokens: int
        cached_tokens: int
        cached_tokens_details: RealtimeModelMetrics.CachedTokenDetails | None

    class OutputTokenDetails(BaseModel):
        text_tokens: int
        audio_tokens: int
        image_tokens: int

    type: Literal["realtime_model_metrics"] = "realtime_model_metrics"
    label: str
    request_id: str
    timestamp: float
    """The timestamp of the response creation."""
    duration: float
    """The duration of the response from created to done in seconds."""
    ttft: float
    """Time to first audio token in seconds. -1 if no audio token was sent."""
    cancelled: bool
    """Whether the request was cancelled."""
    input_tokens: int
    """The number of input tokens used in the Response, including text and audio tokens."""
    output_tokens: int
    """The number of output tokens sent in the Response, including text and audio tokens."""
    total_tokens: int
    """The total number of tokens in the Response."""
    tokens_per_second: float
    """The number of tokens per second."""
    input_token_details: InputTokenDetails
    """Details about the input tokens used in the Response."""
    output_token_details: OutputTokenDetails
    """Details about the output tokens used in the Response."""


class ResponseLatencyMetrics(BaseModel):
    type: Literal["response_latency_metrics"] = "response_latency_metrics"
    timestamp: float
    """When the metric was captured."""
    speech_id: str | None = None
    """The speech ID this latency measurement is associated with."""
    
    # Core end-to-end latency
    e2e_latency: float
    """Total time from end of user speech to first audio response in seconds."""
    
    # Boundary timestamps for verification
    eou_timestamp: float
    """Timestamp when user stopped speaking (from VAD)."""
    first_audio_timestamp: float
    """Timestamp when first audio response was generated."""


class AgentLLMMetrics(BaseModel):
    type: Literal["agent_llm_metrics"] = "agent_llm_metrics"
    timestamp: float
    """When the metric was captured."""
    speech_id: str | None = None
    """The speech ID this LLM processing is associated with."""
    
    # Streaming-aware timing for E2E calculation
    agent_ttft: float | None = None
    """Time from agent start to first token available for TTS in seconds."""
    
    # LLM node execution timing
    llm_node_await: float | None = None
    """Time spent executing the LLM node in seconds."""
    
    # Context for understanding the processing
    request_id: str | None = None
    """Request ID from the underlying LLM provider."""


class ToolExecutionMetrics(BaseModel):
    type: Literal["tool_execution_metrics"] = "tool_execution_metrics"
    timestamp: float
    """When the metric was captured."""
    speech_id: str | None = None
    """The speech ID this tool execution is associated with."""
    
    # Tool execution timing
    total_execution_time: float
    """Total time for all tool executions in seconds."""
    
    # Individual tool durations (detailed breakdown)
    tool_durations: dict[str, float] = {}
    """Per-tool execution durations in seconds."""


class TurnDetectionMetrics(BaseModel):
    type: Literal["turn_detection"] = "turn_detection"
    timestamp: float
    """When the turn detection was performed."""
    
    # Core prediction data
    probability: float
    """Turn detection probability (0.0-1.0) from the model."""
    
    turn_ended: bool
    """Final boolean decision: True if turn should end, False if user likely to continue speaking."""
    
    inference_time: float
    """Time taken for turn detection inference in seconds."""
    
    # Behavioral data
    endpointing_delay: float
    """Actual endpointing delay applied (may be adapted based on collision detection)."""
    
    collision_multiplier: float = 1.0
    """Adaptive endpointing multiplier based on collision patterns."""
    
    # Correlation
    speech_id: str | None = None
    """Links this turn detection to the resulting speech generation."""


AgentMetrics = Union[
    STTMetrics,
    LLMMetrics,
    TTSMetrics,
    VADMetrics,
    EOUMetrics,
    TurnDetectionMetrics,
    RealtimeModelMetrics,
    ResponseLatencyMetrics,
    AgentLLMMetrics,
    ToolExecutionMetrics,
]
