"""Public contracts required by the vox.ai agent-server fork consumer."""

import inspect

import pytest

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.metrics import AgentLLMMetrics, ResponseLatencyMetrics, ToolExecutionMetrics
from livekit.agents.tts import AudioEmitter
from livekit.agents.voice.events import MetricsCollectedEvent, PreemptiveGenerationOutcomeEvent

pytestmark = pytest.mark.unit


async def test_session_accepts_and_updates_custom_options():
    session = AgentSession(
        vad=None,
        interruption_ignore_words=["네"],
        enable_dynamic_interruption=True,
        conversation_continuity_threshold=2.0,
        enable_adaptive_endpointing=False,
        turn_handling={"interruption": {"backchannel_boundary": None}},
    )
    try:
        assert session.options.interruption_ignore_words == ["네"]
        assert session.options.enable_dynamic_interruption is True
        assert session.options.conversation_continuity_threshold == 2.0
        assert session.options.enable_adaptive_endpointing is False
        assert session.options.interruption["backchannel_boundary"] is None
        session.update_options(
            interruption_ignore_words=["예"],
            enable_dynamic_interruption=False,
            conversation_continuity_threshold=3.0,
            enable_adaptive_endpointing=True,
        )
        assert session.options.interruption_ignore_words == ["예"]
        assert session.options.enable_dynamic_interruption is False
        assert session.options.conversation_continuity_threshold == 3.0
        assert session.options.enable_adaptive_endpointing is True
    finally:
        await session.aclose()


def test_metrics_event_roundtrip_preserves_custom_types():
    metrics = [
        AgentLLMMetrics(timestamp=1, agent_ttft=0.1),
        ResponseLatencyMetrics(
            timestamp=1, e2e_latency=0.5, eou_timestamp=0, first_audio_timestamp=0.5
        ),
        ToolExecutionMetrics(timestamp=1, total_execution_time=0.2, tool_durations={"test": 0.2}),
    ]
    for metric in metrics:
        event = MetricsCollectedEvent(metrics=metric)
        restored = MetricsCollectedEvent.model_validate_json(event.model_dump_json())
        assert type(restored.metrics) is type(metric)
        assert restored.metrics == metric
    assert "trigger_source" in PreemptiveGenerationOutcomeEvent.model_fields


def test_reply_callbacks_can_be_removed():
    agent = Agent(instructions="test")
    calls = []

    def callback(ctx, replies):
        calls.append((ctx, replies))

    ctx = llm.ChatContext.empty()
    reply = ctx.add_message(role="assistant", content="test")
    agent.add_reply_callback(callback)
    agent.reply_callback(ctx, [reply])
    agent.remove_reply_callback(callback)
    agent.reply_callback(ctx, [reply])
    assert len(calls) == 1


def test_tts_frame_size_remains_50ms():
    assert inspect.signature(AudioEmitter.initialize).parameters["frame_size_ms"].default == 50
