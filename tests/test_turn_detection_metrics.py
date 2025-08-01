"""Tests for TurnDetectionMetrics functionality."""

import time
import pytest
from livekit.agents.metrics import TurnDetectionMetrics


class TestTurnDetectionMetrics:
    """Test TurnDetectionMetrics creation and serialization."""

    def test_turn_detection_metrics_creation(self):
        """Test TurnDetectionMetrics creates correctly with all fields."""
        timestamp = time.time()
        metrics = TurnDetectionMetrics(
            timestamp=timestamp,
            probability=0.85,
            turn_ended=True,
            inference_time=0.045,
            endpointing_delay=2.0,
            collision_multiplier=1.5,
            speech_id="speech_123"
        )
        
        assert metrics.type == "turn_detection"
        assert metrics.timestamp == timestamp
        assert metrics.probability == 0.85
        assert metrics.turn_ended is True
        assert metrics.inference_time == 0.045
        assert metrics.endpointing_delay == 2.0
        assert metrics.collision_multiplier == 1.5
        assert metrics.speech_id == "speech_123"

    def test_turn_detection_metrics_minimal_creation(self):
        """Test TurnDetectionMetrics creates with minimal required fields."""
        timestamp = time.time()
        metrics = TurnDetectionMetrics(
            timestamp=timestamp,
            probability=0.75,
            turn_ended=False,
            inference_time=0.032,
            endpointing_delay=1.8
        )
        
        assert metrics.type == "turn_detection"
        assert metrics.probability == 0.75
        assert metrics.turn_ended is False
        assert metrics.inference_time == 0.032
        assert metrics.endpointing_delay == 1.8
        assert metrics.collision_multiplier == 1.0  # default value
        assert metrics.speech_id is None  # default value

    def test_turn_detection_metrics_serialization(self):
        """Test TurnDetectionMetrics serializes/deserializes correctly."""
        original = TurnDetectionMetrics(
            timestamp=time.time(),
            probability=0.92,
            turn_ended=True,
            inference_time=0.028,
            endpointing_delay=3.2,
            collision_multiplier=2.1,
            speech_id="test_speech"
        )
        
        # Serialize to dict
        serialized = original.model_dump()
        assert serialized["type"] == "turn_detection"
        assert serialized["probability"] == 0.92
        assert serialized["turn_ended"] is True
        assert serialized["speech_id"] == "test_speech"
        
        # Deserialize back
        deserialized = TurnDetectionMetrics.model_validate(serialized)
        assert deserialized.type == original.type
        assert deserialized.probability == original.probability
        assert deserialized.turn_ended == original.turn_ended
        assert deserialized.inference_time == original.inference_time
        assert deserialized.collision_multiplier == original.collision_multiplier

    def test_turn_detection_metrics_in_agent_metrics_union(self):
        """Test TurnDetectionMetrics is properly included in AgentMetrics union."""
        from livekit.agents.metrics import AgentMetrics
        
        metrics = TurnDetectionMetrics(
            timestamp=time.time(),
            probability=0.88,
            inference_time=0.051,
            endpointing_delay=2.5
        )
        
        # Should be able to assign to AgentMetrics type
        agent_metrics: AgentMetrics = metrics
        assert agent_metrics.type == "turn_detection"

    def test_turn_detection_metrics_validation(self):
        """Test TurnDetectionMetrics field validation."""
        timestamp = time.time()
        
        # Valid probability range (should work)
        metrics = TurnDetectionMetrics(
            timestamp=timestamp,
            probability=0.0,  # minimum valid
            inference_time=0.001,
            endpointing_delay=0.5
        )
        assert metrics.probability == 0.0
        
        metrics = TurnDetectionMetrics(
            timestamp=timestamp,
            probability=1.0,  # maximum valid
            inference_time=0.001,
            endpointing_delay=0.5
        )
        assert metrics.probability == 1.0
        
        # Negative inference time should be possible (edge case handling)
        metrics = TurnDetectionMetrics(
            timestamp=timestamp,
            probability=0.7,
            inference_time=0.0,  # zero is valid
            endpointing_delay=1.0
        )
        assert metrics.inference_time == 0.0

    def test_turn_detection_metrics_defaults(self):
        """Test TurnDetectionMetrics default values."""
        metrics = TurnDetectionMetrics(
            timestamp=time.time(),
            probability=0.6,
            inference_time=0.04,
            endpointing_delay=1.5
            # collision_multiplier and speech_id not provided
        )
        
        assert metrics.collision_multiplier == 1.0
        assert metrics.speech_id is None

if __name__ == "__main__":
    pytest.main([__file__])