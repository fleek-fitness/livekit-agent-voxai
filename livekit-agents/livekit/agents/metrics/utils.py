from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import (
    AgentLLMMetrics,
    AgentMetrics,
    EOUMetrics,
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

    if isinstance(metrics, LLMMetrics):
        logger.info(
            "LLM metrics",
            extra={
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
            extra={
                "ttft": round(metrics.ttft, 2),
                "input_tokens": metrics.input_tokens,
                "cached_input_tokens": metrics.input_token_details.cached_tokens,
                "output_tokens": metrics.output_tokens,
                "total_tokens": metrics.total_tokens,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
            },
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            "TTS metrics",
            extra={
                "ttfb": metrics.ttfb,
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            "EOU metrics",
            extra={
                "end_of_utterance_delay": round(metrics.end_of_utterance_delay, 2),
                "transcription_delay": round(metrics.transcription_delay, 2),
            },
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(
            "STT metrics",
            extra={
                "audio_duration": round(metrics.audio_duration, 2),
            },
        )
    elif isinstance(metrics, ResponseLatencyMetrics):
        logger.info(f"Response Latency metrics: end_to_end={metrics.e2e_latency:.2f}s")
    elif isinstance(metrics, AgentLLMMetrics):
        agent_ttft_str = (
            f"agent_ttft={metrics.agent_ttft:.2f}s" if metrics.agent_ttft else "agent_ttft=None"
        )
        logger.info(
            f"Agent LLM metrics: {agent_ttft_str}"  # noqa: E501
        )
    elif isinstance(metrics, ToolExecutionMetrics):
        tool_count = len(metrics.tool_durations)
        tool_names = list(metrics.tool_durations.keys())
        logger.info(
            f"Tool Execution metrics: total_time={metrics.total_execution_time:.2f}s, tools_executed={tool_count}, tools={tool_names}"  # noqa: E501
        )
