from __future__ import annotations

import logging

from ..log import logger as default_logger
from .base import (
    AgentMetrics,
    AgentLLMMetrics,
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
            f"LLM metrics: ttft={metrics.ttft:.2f}, input_tokens={metrics.prompt_tokens},  cached_input_tokens={metrics.prompt_cached_tokens}, output_tokens={metrics.completion_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, RealtimeModelMetrics):
        logger.info(
            f"RealtimeModel metrics: ttft={metrics.ttft:.2f}, input_tokens={metrics.input_tokens}, cached_input_tokens={metrics.input_token_details.cached_tokens}, output_tokens={metrics.output_tokens}, total_tokens={metrics.total_tokens}, tokens_per_second={metrics.tokens_per_second:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, TTSMetrics):
        logger.info(
            f"TTS metrics: ttfb={metrics.ttfb}, audio_duration={metrics.audio_duration:.2f}"
        )
    elif isinstance(metrics, EOUMetrics):
        logger.info(
            f"EOU metrics: end_of_utterance_delay={metrics.end_of_utterance_delay:.2f}, transcription_delay={metrics.transcription_delay:.2f}"  # noqa: E501
        )
    elif isinstance(metrics, STTMetrics):
        logger.info(f"STT metrics: audio_duration={metrics.audio_duration:.2f}")
    elif isinstance(metrics, ResponseLatencyMetrics):
        logger.info(
            f"Response Latency metrics: end_to_end={metrics.end_to_end_latency:.2f}s"
        )
    elif isinstance(metrics, AgentLLMMetrics):
        logger.info(
            f"Agent LLM metrics: llm_node_duration={metrics.llm_node_duration:.2f}s"  # noqa: E501
        )
    elif isinstance(metrics, ToolExecutionMetrics):
        tool_count = len(metrics.individual_durations)
        tool_names = list(metrics.individual_durations.keys())
        logger.info(
            f"Tool Execution metrics: total_time={metrics.total_execution_time:.2f}s, tools_executed={tool_count}, tools={tool_names}"  # noqa: E501
        )
