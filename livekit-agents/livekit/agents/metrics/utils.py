from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import (
    AgentLLMMetrics,
    AgentMetrics,
    AvatarMetrics,
    EOUMetrics,
    InterruptionMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    ResponseLatencyMetrics,
    STTMetrics,
    ToolExecutionMetrics,
    TTSMetrics,
)


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = default_logger

    metadata: dict[str, object] = {}
    if metrics.metadata:
        metadata |= {
            "model_name": metrics.metadata.model_name or "unknown",
            "model_provider": metrics.metadata.model_provider or "unknown",
        }

    if isinstance(metrics, LLMMetrics):
        logger.info(
            "LLM metrics",
            extra=metadata
            | {
                "ttft": round(metrics.ttft, 2),
                "prompt_tokens": metrics.prompt_tokens,
                "prompt_cached_tokens": metrics.prompt_cached_tokens,
                "completion_tokens": metrics.completion_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, RealtimeModelMetrics):
        logger.info(
            "RealtimeModel metrics",
            extra=metadata
            | {
                "ttft": round(metrics.ttft, 2),
                "input_tokens": metrics.input_tokens,
                "cached_input_tokens": metrics.input_token_details.cached_tokens,
                "input_text_tokens": metrics.input_token_details.text_tokens,
                "input_cached_text_tokens": metrics.input_token_details.cached_tokens_details.text_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "input_image_tokens": metrics.input_token_details.image_tokens,
                "input_cached_image_tokens": metrics.input_token_details.cached_tokens_details.image_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "input_audio_tokens": metrics.input_token_details.audio_tokens,
                "input_cached_audio_tokens": metrics.input_token_details.cached_tokens_details.audio_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0,
                "output_tokens": metrics.output_tokens,
                "output_text_tokens": metrics.output_token_details.text_tokens,
                "output_audio_tokens": metrics.output_token_details.audio_tokens,
                "output_image_tokens": metrics.output_token_details.image_tokens,
                "total_tokens": metrics.total_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            "TTS metrics",
            extra=metadata
            | {
                "ttfb": metrics.ttfb,
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            "EOU metrics",
            extra=metadata
            | {
                "end_of_utterance_delay": round(metrics.end_of_utterance_delay, 2),
                "transcription_delay": round(metrics.transcription_delay, 2),
            },
        )
    elif isinstance(metrics, ResponseLatencyMetrics):
        logger.info(
            "Response latency metrics",
            extra=metadata
            | {
                "speech_id": metrics.speech_id or "",
                "e2e_latency": round(metrics.e2e_latency, 2),
                "eou_timestamp": metrics.eou_timestamp,
                "first_audio_timestamp": metrics.first_audio_timestamp,
            },
        )
    elif isinstance(metrics, AgentLLMMetrics):
        logger.info(
            "Agent LLM metrics",
            extra=metadata
            | {
                "speech_id": metrics.speech_id or "",
                "agent_ttft": round(metrics.agent_ttft, 2)
                if metrics.agent_ttft is not None
                else None,
                "llm_node_await": round(metrics.llm_node_await, 2)
                if metrics.llm_node_await is not None
                else None,
                "request_id": metrics.request_id or "",
            },
        )
    elif isinstance(metrics, ToolExecutionMetrics):
        logger.info(
            "Tool execution metrics",
            extra=metadata
            | {
                "speech_id": metrics.speech_id or "",
                "total_execution_time": round(metrics.total_execution_time, 2),
                "tool_durations": {
                    name: round(duration, 2) for name, duration in metrics.tool_durations.items()
                },
            },
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(
            "STT metrics",
            extra=metadata
            | {
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, InterruptionMetrics):
        logger.info(
            "Interruption metrics",
            extra=metadata
            | {
                "total_duration": round(metrics.total_duration, 2),
                "prediction_duration": round(metrics.prediction_duration, 2),
                "detection_delay": round(metrics.detection_delay, 2),
                "num_interruptions": metrics.num_interruptions,
                "num_backchannels": metrics.num_backchannels,
                "num_requests": metrics.num_requests,
            },
        )
    elif isinstance(metrics, AvatarMetrics):
        extra: dict[str, str | float] = {}
        if metrics.session_started_time and metrics.avatar_joined_time:
            extra["avatar_join_latency"] = round(
                metrics.avatar_joined_time - metrics.session_started_time, 3
            )
        if metrics.playback_latency:
            extra["playback_latency"] = round(metrics.playback_latency, 3)
        logger.info("Avatar metrics", extra=metadata | extra)
