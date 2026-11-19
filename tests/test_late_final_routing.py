"""Routing of a final that reaches AudioRecognition after its turn, and the record queue.

AudioRecognition decides from spans and onsets alone whether a final belongs to an already
committed turn; the RecordGate keeps the stored order while such a turn is open. Both are driven
directly here with the exact event orders the provider shapes produce.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition, _ConfirmedTurn
from livekit.agents.voice.record_gate import RecordGate

from .test_held_segment_turn_owner import _recognition

pytestmark = pytest.mark.unit


def _final(text: str, *, span: Any = None, onset: float | None = None) -> SpeechEvent:
    kwargs: dict[str, Any] = {} if span == "not_given" else {"transcript_span": span}
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text=text)],
        finalize_onset=onset,
        **kwargs,
    )


def _with_turn(ar: AudioRecognition) -> AudioRecognition:
    ar._confirmed_turns = {5: _ConfirmedTurn(5, [(100.0, 101.0), (101.4, 102.0)], 102.0)}
    ar._completed_vad_bursts = deque([(130.0, 131.0)])
    ar._active_vad_speech_started_at = None
    ar._last_confirmed_end = 102.0
    return ar


async def _send(ar: AudioRecognition, ev: SpeechEvent) -> tuple[list[Any], int]:
    await ar._on_stt_event(ev)
    late = [(c.kwargs["turn_id"], c.kwargs["span"]) for c in ar._hooks.on_late_final.call_args_list]
    return late, ar._process_stt_event.call_count


async def test_span_inside_a_turn_segment_is_that_turns_late_final() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("없습니다", span=(100.5, 100.9)))
    assert late == [(5, (100.5, 101.0))]
    assert normal == 0


async def test_span_over_the_gap_between_segments_is_not_that_turns() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("음", span=(101.1, 101.3)))
    # not the turn's audio, but it ended before everything the pending turn has seen:
    # recorded as its own message where it arrives, never joined to the pending turn
    assert late == [(None, (101.1, 101.3))]
    assert normal == 0


async def test_onset_without_span_takes_the_segment_of_the_turn() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("없습니다", span=None, onset=101.4))
    assert late == [(5, (101.4, 102.0))]
    assert normal == 0


async def test_no_span_and_no_onset_keeps_the_normal_path() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("없습니다", span=None))
    assert late == []
    assert normal == 1


async def test_unknown_onset_keeps_the_normal_path() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("없습니다", span=None, onset=555.0))
    assert late == []
    assert normal == 1


async def test_provider_without_spans_keeps_the_normal_path() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("없습니다", span="not_given"))
    assert late == []
    assert normal == 1


async def test_final_of_the_open_segment_is_not_late() -> None:
    ar = _with_turn(_recognition(aligned=False))
    ar._active_vad_speech_started_at = 129.0
    late, normal = await _send(ar, _final("네", span=(129.4, 129.9)))
    assert late == []
    assert normal == 1


async def test_final_of_the_pending_turn_is_not_late() -> None:
    ar = _with_turn(_recognition(aligned=False))
    late, normal = await _send(ar, _final("네", span=(130.2, 130.8)))
    assert late == []
    assert normal == 1


async def test_record_gone_but_earlier_than_the_pending_turn_is_recorded_alone() -> None:
    ar = _with_turn(_recognition(aligned=False))
    ar._confirmed_turns = {}
    late, normal = await _send(ar, _final("없습니다", span=(100.5, 100.9)))
    assert late == [(None, (100.5, 100.9))]
    assert normal == 0


async def test_watermark_closes_turns_and_a_closed_turn_is_forgotten() -> None:
    ar = _with_turn(_recognition(aligned=False))
    await ar._on_stt_event(SpeechEvent(type=SpeechEventType.FINALIZED_UNTIL, speech_end_time=101.5))
    assert ar._confirmed_turns == {5: ar._confirmed_turns[5]}  # audio up to 102.0 still open
    await ar._on_stt_event(SpeechEvent(type=SpeechEventType.FINALIZED_UNTIL, speech_end_time=102.0))
    assert ar._confirmed_turns == {}
    ar._hooks.on_turn_released.assert_called_once_with(5)
    # a closed turn takes no late final
    late, normal = await _send(ar, _final("없습니다", span=(100.5, 100.9)))
    assert late == [(None, (100.5, 100.9))]


async def test_turn_committed_after_the_watermark_is_never_held() -> None:
    ar = _recognition(aligned=False)
    ar._stt_watermark = 200.0
    info = SimpleNamespace(turn_id=9, normal_turn=True, transcript_span=(150.0, 151.0))
    ar._record_confirmed_turn(info, [(150.0, 151.0)], [(150.2, 150.9)])  # type: ignore[arg-type]
    assert not ar._confirmed_turns
    ar._hooks.on_turn_released.assert_called_once_with(9)


async def test_backchannel_turn_and_unsupported_provider_are_not_recorded() -> None:
    ar = _recognition(aligned=False)
    ar._record_confirmed_turn(
        SimpleNamespace(turn_id=1, normal_turn=False, transcript_span=(1.0, 2.0)),  # type: ignore[arg-type]
        [(1.0, 2.0)],
        [],
    )
    from livekit.agents import NOT_GIVEN

    ar._record_confirmed_turn(
        SimpleNamespace(turn_id=2, normal_turn=True, transcript_span=NOT_GIVEN),  # type: ignore[arg-type]
        [(1.0, 2.0)],
        [],
    )
    assert not ar._confirmed_turns


async def test_empty_signal_never_erases_a_segment_that_has_text() -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(10.0, 11.0)])
    ar._turn_final_spans = [(10.2, 10.9)]
    await ar._on_stt_event(
        SpeechEvent(type=SpeechEventType.VAD_FINALIZE_EMPTY, finalize_onset=10.0)
    )
    assert list(ar._completed_vad_bursts) == [(10.0, 11.0)]


async def test_empty_signal_closes_only_the_named_segment() -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(10.0, 10.3), (12.0, 13.0)])
    await ar._on_stt_event(
        SpeechEvent(type=SpeechEventType.VAD_FINALIZE_EMPTY, finalize_onset=10.0)
    )
    assert list(ar._completed_vad_bursts) == [(12.0, 13.0)]
    await ar._on_stt_event(
        SpeechEvent(type=SpeechEventType.VAD_FINALIZE_EMPTY, finalize_onset=99.0)
    )
    assert list(ar._completed_vad_bursts) == [(12.0, 13.0)]


async def test_a_span_proves_earlier_textless_segments_empty() -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(1.0, 1.3), (5.0, 6.0), (8.0, 9.0)])
    ar._turn_final_spans = [(5.1, 5.8)]
    ar._close_empty_bursts_before((8.2, 8.9))
    # the first segment had no text; the second kept its own; the third is this final's
    assert list(ar._completed_vad_bursts) == [(5.0, 6.0), (8.0, 9.0)]


def test_onset_names_a_segment_only_when_it_is_unique() -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(10.0, 11.0), (12.0, 13.0)])
    assert ar._resolve_final_span(_final("a", span=None, onset=12.0)) == (12.0, 13.0)
    assert ar._resolve_final_span(_final("a", span=None, onset=7.0)) is None
    ar._completed_vad_bursts = deque([(10.0, 11.0), (10.0, 12.0)])
    assert ar._resolve_final_span(_final("a", span=None, onset=10.0)) is None
    # a span the provider gave is never replaced
    assert ar._resolve_final_span(_final("a", span=(1.0, 2.0), onset=10.0)) == (1.0, 2.0)


# --- the record queue -------------------------------------------------------------------------


def _gate() -> tuple[RecordGate, list[str], Any]:
    out: list[str] = []
    gate = RecordGate(lambda name, arg: out.append(arg.item))

    def event(label: str, bound_to: int | None = None) -> Any:
        ev = SimpleNamespace(item=label)
        return ev

    return gate, out, event


async def test_late_final_is_recorded_after_its_turn_and_before_the_reply() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    # the turn's message, then the reply to it, arrive in order
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    out_len_after_message = len(out)
    gate.push("conversation_item_added", event("reply"))
    assert out_len_after_message == 1  # the turn's own message is never held
    assert out[1:] == []  # the reply waits behind the open turn
    gate.insert_late(1, "conversation_item_added", event("late"))
    assert out[1:] == ["late"]
    gate.release_turn(1)
    assert out[1:] == ["late", "reply"]


async def test_other_records_keep_their_relative_order_behind_an_open_turn() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    for label in ("reply", "tool call", "tool output", "next question"):
        gate.push("conversation_item_added", event(label))
    gate.release_turn(1)
    assert out[1:] == ["reply", "tool call", "tool output", "next question"]


async def test_two_open_turns_release_in_turn_order() -> None:
    gate, out, event = _gate()
    m1, m2 = SimpleNamespace(), SimpleNamespace()
    gate.open_turn(1)
    gate.bind_message(1, m1)
    gate.open_turn(2)
    gate.bind_message(2, m2)
    gate.push("conversation_item_added", SimpleNamespace(item=m1))
    gate.push("conversation_item_added", event("reply 1"))
    gate.push("conversation_item_added", SimpleNamespace(item=m2))
    gate.push("conversation_item_added", event("reply 2"))
    gate.insert_late(2, "conversation_item_added", event("late 2"))
    gate.insert_late(1, "conversation_item_added", event("late 1"))
    assert out == [m1, "late 1"]
    gate.release_turn(2)  # the newer turn closing first changes nothing for the older one
    assert out == [m1, "late 1"]
    gate.release_turn(1)
    assert out == [m1, "late 1", "reply 1", m2, "late 2", "reply 2"]


async def test_close_flushes_everything_once_and_a_late_record_after_release_is_emitted_now() -> (
    None
):
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    gate.push("conversation_item_added", event("reply"))
    gate.release_all()
    assert out == [message, "reply"]
    gate.insert_late(1, "conversation_item_added", event("very late"))
    assert out == [message, "reply", "very late"]
    gate.release_all()
    assert out == [message, "reply", "very late"]


async def test_late_final_before_the_turn_message_follows_it() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.insert_late(1, "conversation_item_added", event("late"))
    assert out == []
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    gate.push("conversation_item_added", event("reply"))
    gate.release_turn(1)
    assert out == [message, "late", "reply"]


async def test_late_final_is_consumed_even_when_its_record_was_dropped_and_vad_is_close() -> None:
    ar = _recognition(aligned=False)
    ar._last_confirmed_end = 102.0
    ar._completed_vad_bursts = deque([(102.05, 103.0)])
    late, normal = await _send(ar, _final("없습니다", span=(101.5, 101.95)))
    assert late == [(None, (101.5, 101.95))]
    assert normal == 0


async def test_without_a_committed_turn_nothing_is_late() -> None:
    # the first utterance whose VAD segment is missing must still get its reply
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(130.0, 131.0)])
    late, normal = await _send(ar, _final("여보세요", span=(100.0, 100.8)))
    assert late == []
    assert normal == 1


async def test_a_final_after_the_last_committed_turn_is_the_pending_turns() -> None:
    ar = _with_turn(_recognition(aligned=False))
    ar._confirmed_turns = {}
    late, normal = await _send(ar, _final("네", span=(102.5, 102.9)))
    assert late == []
    assert normal == 1


async def test_empty_segment_clears_the_clock_it_set_through_an_inference_event() -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(10.0, 10.3), (12.0, 13.0)])
    ar._speech_start_time = 10.07  # set by INFERENCE_DONE, not equal to the SOS time
    ar._speech_clock_burst = 10.0
    ar._retire_empty_vad_burst(10.0)
    assert ar._speech_start_time == 12.0


@pytest.mark.parametrize("first", [1.0, 2.0])
async def test_overlap_flag_stays_up_until_every_waiting_segment_is_gone(first: float) -> None:
    ar = _recognition(aligned=False)
    ar._completed_vad_bursts = deque([(1.0, 1.3), (2.0, 3.0)])
    ar._agent_overlap_segment_unfinalized = True
    ar._overlap_unfinalized_bursts = frozenset({1.0, 2.0})
    ar._retire_empty_vad_burst(first)
    assert ar._agent_overlap_segment_unfinalized is True
    ar._retire_empty_vad_burst(3.0 - first)
    assert ar._agent_overlap_segment_unfinalized is False


async def test_a_final_clears_the_overlap_flag_and_its_segments() -> None:
    ar = _recognition(aligned=False)
    ar._agent_overlap_segment_unfinalized = True
    ar._overlap_unfinalized_bursts = frozenset({1.0})
    ar._agent_overlap_segment_unfinalized = False
    assert ar._overlap_unfinalized_bursts == frozenset()


async def test_provider_without_spans_never_runs_the_span_corrections() -> None:
    ar = _recognition(aligned=False)
    ar._process_stt_event = AudioRecognition._process_stt_event.__get__(ar)  # type: ignore[method-assign]
    ar._session.options.turn_handling = {"user_turn_limit": {}}
    ar._anchor_turn_to_first_final = MagicMock()  # type: ignore[method-assign]
    ar._close_empty_bursts_before = MagicMock()  # type: ignore[method-assign]
    ar._completed_vad_bursts = deque([(1.0, 1.3), (5.0, 6.0)])
    ar._user_turn_start = 1.0
    await ar._on_stt_event(_final("네", span="not_given"))
    ar._anchor_turn_to_first_final.assert_not_called()
    ar._close_empty_bursts_before.assert_not_called()


# --- the record queue: the turn's own message is added by a task ------------------------------


async def test_late_final_waits_for_a_message_that_is_not_there_yet_after_release() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.insert_late(1, "conversation_item_added", event("second"))
    gate.release_turn(1)  # the watermark closed the turn before the first message was added
    assert out == []
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    gate.push("conversation_item_added", event("reply"))
    assert out == [message, "second", "reply"]


async def test_session_end_emits_a_late_final_whose_message_never_came_once() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    gate.bind_message(1, SimpleNamespace())
    gate.insert_late(1, "conversation_item_added", event("second"))
    gate.release_all()
    gate.release_all()
    assert out == ["second"]


async def test_session_end_orders_first_then_second_once() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.insert_late(1, "conversation_item_added", event("second"))
    gate.push("conversation_item_added", SimpleNamespace(item=message))
    gate.release_all()
    assert out == [message, "second"]


async def test_the_session_emit_path_hands_the_waiting_first_message_to_the_gate() -> None:
    gate, out, event = _gate()
    gate.open_turn(1)
    message = SimpleNamespace()
    gate.bind_message(1, message)
    gate.insert_late(1, "conversation_item_added", event("second"))
    gate.release_turn(1)  # closed by the watermark before the first message exists

    def emit(name: str, arg: Any) -> None:  # what AgentSession.emit does
        if gate.active:
            gate.push(name, arg)
        else:
            out.append(arg.item)

    emit("conversation_item_added", SimpleNamespace(item=message))
    emit("conversation_item_added", event("reply"))
    assert out == [message, "second", "reply"]  # in order, before the session ends
    gate.release_all()
    assert out == [message, "second", "reply"]


# --- the end of a message stretches to the end of its VAD segment -------------------------------


def test_end_stretches_to_the_one_overlapping_segment() -> None:
    from livekit.agents.voice.audio_recognition import _extend_span_end

    assert _extend_span_end((10.0, 12.0), [(9.9, 12.6)]) == (10.0, 12.6)


def test_end_never_stretches_past_one_second_or_shrinks() -> None:
    from livekit.agents.voice.audio_recognition import _extend_span_end

    assert _extend_span_end((10.0, 12.0), [(9.9, 13.5)]) == (10.0, 12.0)  # trailing noise
    assert _extend_span_end((10.0, 12.0), [(9.9, 12.0)]) == (10.0, 12.0)
    assert _extend_span_end((10.0, 12.0), [(9.9, 11.5)]) == (10.0, 12.0)  # never shrinks
    assert _extend_span_end((10.0, 12.0), [(9.9, 13.0)]) == (10.0, 13.0)  # exactly 1.0 allowed


def test_end_is_kept_with_zero_or_several_overlapping_segments() -> None:
    from livekit.agents.voice.audio_recognition import _extend_span_end

    assert _extend_span_end((10.0, 12.0), [(20.0, 21.0)]) == (10.0, 12.0)
    assert _extend_span_end((10.0, 12.0), [(9.9, 11.0), (11.2, 12.5)]) == (10.0, 12.0)


async def test_late_final_end_follows_the_same_rule() -> None:
    ar = _with_turn(_recognition(aligned=False))
    ar._confirmed_turns[5].segments[0] = (100.0, 101.4)
    late, _ = await _send(ar, _final("없습니다", span=(100.5, 101.0)))
    assert late == [(5, (100.5, 101.4))]
