from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, ClassVar, Generic, Literal, TypeVar, Union

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc
from livekit.agents.metrics.base import Metadata

from .. import utils
from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..metrics import LLMMetrics
from ..telemetry import _chat_ctx_to_otel_events, trace_types, tracer, utils as telemetry_utils
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio
from .chat_context import ChatContext, ChatRole
from .tool_context import FunctionTool, ProviderTool, RawFunctionTool, ToolChoice


class CompletionUsage(BaseModel):
    completion_tokens: int
    """The number of tokens in the completion."""
    prompt_tokens: int
    """The number of input tokens used (includes cached tokens)."""
    prompt_cached_tokens: int = 0
    """The number of cached input tokens used."""
    cache_creation_tokens: int = 0
    """The number of tokens used to create the cache."""
    cache_read_tokens: int = 0
    """The number of tokens read from the cache."""
    total_tokens: int
    """The total number of tokens used (completion + prompt tokens)."""


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str
    extra: dict[str, Any] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class ChoiceDelta(BaseModel):
    role: ChatRole | None = None
    content: str | None = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)
    extra: dict[str, Any] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class ChatChunk(BaseModel):
    id: str
    delta: ChoiceDelta | None = None
    usage: CompletionUsage | None = None


class LLMError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["llm_error"] = "llm_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class LLM(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]],
    Generic[TEvent],
):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        """Get the model name/identifier for this LLM instance.

        Returns:
            The model name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their model information.
        """
        return "unknown"

    @property
    def provider(self) -> str:
        """Get the provider name/identifier for this LLM instance.

        Returns:
            The provider name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their provider information.
        """
        return "unknown"

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool | ProviderTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream: ...

    def prewarm(self) -> None:
        """Pre-warm connection to the LLM service"""
        pass

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class LLMStream(ABC):
    _llm_request_span_name: ClassVar[str] = "llm_request"

    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool | ProviderTool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options
        self.__fnc_name: str | None = None
        self._tool_name_callbacks: list[Callable[[str], None]] = []
        self._detected_tool_keys: set[tuple[str, str | None]] = set()

        self._event_ch = aio.Chan[ChatChunk]()
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._current_attempt_has_error = False
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="LLM._metrics_task"
        )

        async def _traceable_main_task() -> None:
            with tracer.start_as_current_span(
                self._llm_request_span_name, end_on_exit=False
            ) as span:
                for name, attributes in _chat_ctx_to_otel_events(self._chat_ctx):
                    span.add_event(name, attributes)
                await self._main_task()

        self._task = asyncio.create_task(_traceable_main_task(), name="LLM._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._llm_request_span: trace.Span | None = None

    def on_tool_name_detected(self, callback: Callable[[str], None]) -> None:
        """Register callback for early tool name detection in this stream."""
        self._tool_name_callbacks.append(callback)

    @property
    def _fnc_name(self) -> str | None:
        return self.__fnc_name

    @_fnc_name.setter
    def _fnc_name(self, value: str | None) -> None:
        prev = self.__fnc_name
        self.__fnc_name = value

        # `_fnc_name` is used by OpenAI-compatible/Anthropic/AWS plugins.
        # Fire only on first observation for this value transition.
        if value and value != prev:
            call_id = getattr(self, "_tool_call_id", None)
            if not isinstance(call_id, str) or not call_id:
                call_id = None

            logger.debug(
                "[EARLY_SPEAK_TRACE] tool name observed via _fnc_name | name=%s call_id=%s",
                value,
                call_id,
            )
            self._fire_tool_name_detected(value, call_id=call_id)

    def _fire_tool_name_detected(
        self,
        name: str,
        *,
        call_id: str | None = None,
    ) -> None:
        if not self._tool_name_callbacks or not name:
            return

        dedupe_key = (name, call_id)
        if dedupe_key in self._detected_tool_keys:
            logger.debug(
                "[EARLY_SPEAK_TRACE] duplicate tool name ignored | name=%s call_id=%s",
                name,
                call_id,
            )
            return
        self._detected_tool_keys.add(dedupe_key)

        logger.debug(
            "[EARLY_SPEAK_TRACE] tool name callback fired | name=%s call_id=%s callbacks=%s",
            name,
            call_id,
            len(self._tool_name_callbacks),
        )
        for cb in self._tool_name_callbacks:
            try:
                cb(name)
            except Exception:
                logger.warning("tool_name_detected callback failed", exc_info=True)

    def _detect_tool_names_from_chunk(self, chunk: ChatChunk) -> None:
        # Google/Mistral style: direct tool_calls emission path.
        if not self._tool_name_callbacks or chunk.delta is None or not chunk.delta.tool_calls:
            return

        for tool_call in chunk.delta.tool_calls:
            if not tool_call.name:
                continue

            call_id = tool_call.call_id if tool_call.call_id else None
            logger.debug(
                "[EARLY_SPEAK_TRACE] tool name observed via chunk delta | name=%s call_id=%s",
                tool_call.name,
                call_id,
            )
            self._fire_tool_name_detected(tool_call.name, call_id=call_id)

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        self._llm_request_span = trace.get_current_span()
        self._llm_request_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, self._llm.model)

        for i in range(self._conn_options.max_retry + 1):
            try:
                with tracer.start_as_current_span("llm_request_run") as attempt_span:
                    attempt_span.set_attribute(trace_types.ATTR_RETRY_COUNT, i)
                    try:
                        return await self._run()
                    except Exception as e:
                        telemetry_utils.record_exception(attempt_span, e)
                        raise
            except APIError as e:
                retry_interval = self._conn_options._interval_for_retry(i)

                if self._conn_options.max_retry == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif i == self._conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to generate LLM completion after {self._conn_options.max_retry + 1} attempts",  # noqa: E501
                    ) from e

                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to generate LLM completion, retrying in {retry_interval}s",  # noqa: E501
                        exc_info=e,
                        extra={
                            "llm": self._llm._label,
                            "attempt": i + 1,
                        },
                    )

                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)

                # reset the flag when retrying
                self._current_attempt_has_error = False

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._current_attempt_has_error = True
        self._llm.emit(
            "error",
            LLMError(
                timestamp=time.time(),
                label=self._llm._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    @utils.log_exceptions(logger=logger)
    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[ChatChunk]) -> None:
        start_time = time.perf_counter()
        ttft = -1.0
        request_id = ""
        usage: CompletionUsage | None = None

        response_content = ""
        tool_calls: list[FunctionToolCall] = []
        completion_start_time: str | None = None

        async for ev in event_aiter:
            request_id = ev.id
            if ttft == -1.0:
                ttft = time.perf_counter() - start_time
                completion_start_time = datetime.now(timezone.utc).isoformat()

            if ev.delta:
                if ev.delta.content:
                    response_content += ev.delta.content
                if ev.delta.tool_calls:
                    tool_calls.extend(ev.delta.tool_calls)

            if ev.usage is not None:
                usage = ev.usage

        duration = time.perf_counter() - start_time

        # if generation is aborted before any tokens are received, it doesn't make sense to report -1 ttft
        if self._current_attempt_has_error or ttft < 0:
            return

        metrics = LLMMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttft=ttft,
            duration=duration,
            cancelled=self._task.cancelled(),
            label=self._llm._label,
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            prompt_cached_tokens=usage.prompt_cached_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            tokens_per_second=usage.completion_tokens / duration if usage else 0.0,
            metadata=Metadata(
                model_name=self._llm.model,
                model_provider=self._llm.provider,
            ),
        )
        if self._llm_request_span:
            # livekit metrics attribute
            self._llm_request_span.set_attribute(
                trace_types.ATTR_LLM_METRICS, metrics.model_dump_json()
            )

            # set gen_ai attributes
            self._llm_request_span.set_attributes(
                {
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: metrics.prompt_tokens,
                    trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: metrics.completion_tokens,
                },
            )
            if completion_start_time:
                self._llm_request_span.set_attribute(
                    trace_types.ATTR_LANGFUSE_COMPLETION_START_TIME, f'"{completion_start_time}"'
                )

            completion_event_body: dict[str, AttributeValue] = {"role": "assistant"}
            if response_content:
                completion_event_body["content"] = response_content
            if tool_calls:
                completion_event_body["tool_calls"] = [
                    json.dumps(
                        {
                            "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                            "id": tool_call.call_id,
                            "type": "function",
                        }
                    )
                    for tool_call in tool_calls
                ]
            self._llm_request_span.add_event(trace_types.EVENT_GEN_AI_CHOICE, completion_event_body)

        self._llm.emit("metrics_collected", metrics)

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[FunctionTool | RawFunctionTool | ProviderTool]:
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        await self._metrics_task
        if self._llm_request_span:
            self._llm_request_span.end()
            self._llm_request_span = None

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        self._detect_tool_names_from_chunk(val)
        return val

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __aenter__(self) -> LLMStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def to_str_iterable(self) -> AsyncIterable[str]:
        """
        Convert the LLMStream to an async iterable of strings.
        This assumes the stream will not call any tools.
        """

        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()
