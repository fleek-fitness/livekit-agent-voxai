#!/usr/bin/env python3

"""
Turn Detection Logging Example

This example demonstrates how to use the new TurnDetectionMetrics feature
to log turn detection results and correlate them with speech generation.

Usage:
    python turn_detection_logging_example.py
"""

import asyncio
import logging
import time
from typing import Dict

# Mock imports for demonstration (in real usage, these would be real imports)
# from livekit.agents import AgentSession
# from livekit.agents.metrics import TurnDetectionMetrics
# from livekit.agents.voice import SpeechCreatedEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationTracker:
    """
    Example class showing how to track and correlate turn detection
    results with speech generation for comprehensive conversation logging.
    """
    
    def __init__(self):
        self.turn_data: Dict[str, dict] = {}  # speech_id -> turn detection data
        self.conversation_turns = []  # Complete conversation history
    
    def handle_turn_detection(self, event):
        """Handle turn detection metrics events."""
        if event.metrics.type == "turn_detection":
            turn_metrics = event.metrics
            
            # Store for correlation with speech
            if turn_metrics.speech_id:
                self.turn_data[turn_metrics.speech_id] = {
                    'probability': turn_metrics.probability,
                    'turn_ended': turn_metrics.turn_ended,
                    'inference_time_ms': turn_metrics.inference_time * 1000,
                    'endpointing_delay': turn_metrics.endpointing_delay,
                    'collision_multiplier': turn_metrics.collision_multiplier,
                    'timestamp': turn_metrics.timestamp
                }
            
            # Log individual turn detection result
            logger.info(
                "TURN_DETECTION",
                extra={
                    'probability': turn_metrics.probability,
                    'turn_ended': turn_metrics.turn_ended,
                    'inference_time_ms': turn_metrics.inference_time * 1000,
                    'endpointing_delay': turn_metrics.endpointing_delay,
                    'collision_multiplier': turn_metrics.collision_multiplier,
                    'speech_id': turn_metrics.speech_id,
                    'timestamp': turn_metrics.timestamp
                }
            )
    
    def handle_speech_created(self, event):
        """Handle speech creation events and correlate with turn detection."""
        speech_id = getattr(event.speech_handle, 'id', None)
        
        if speech_id and speech_id in self.turn_data:
            turn_data = self.turn_data[speech_id]
            
            # Create complete conversation turn record
            conversation_turn = {
                'turn_probability': turn_data['probability'],
                'turn_ended': turn_data['turn_ended'],
                'turn_inference_ms': turn_data['inference_time_ms'],
                'endpointing_delay': turn_data['endpointing_delay'],
                'collision_multiplier': turn_data['collision_multiplier'],
                'agent_response_preview': event.speech_handle.text[:100] + "..." if len(event.speech_handle.text) > 100 else event.speech_handle.text,
                'response_latency_ms': (time.time() - turn_data['timestamp']) * 1000,
                'speech_source': event.source,
                'user_initiated': event.user_initiated,
                'timestamp': time.time()
            }
            
            # Log complete conversation turn
            logger.info("CONVERSATION_TURN", extra=conversation_turn)
            
            # Store in conversation history for analytics
            self.conversation_turns.append(conversation_turn)
            
            # Clean up
            del self.turn_data[speech_id]
    
    def get_conversation_analytics(self):
        """Generate analytics from conversation history."""
        if not self.conversation_turns:
            return {}
        
        probabilities = [turn['turn_probability'] for turn in self.conversation_turns]
        inference_times = [turn['turn_inference_ms'] for turn in self.conversation_turns]
        collision_multipliers = [turn['collision_multiplier'] for turn in self.conversation_turns]
        
        return {
            'total_turns': len(self.conversation_turns),
            'avg_turn_probability': sum(probabilities) / len(probabilities),
            'avg_inference_time_ms': sum(inference_times) / len(inference_times),
            'max_inference_time_ms': max(inference_times),
            'avg_collision_multiplier': sum(collision_multipliers) / len(collision_multipliers),
            'high_collision_turns': len([m for m in collision_multipliers if m > 2.0]),
            'low_confidence_turns': len([p for p in probabilities if p < 0.5])
        }


def setup_turn_detection_logging(session):
    """
    Setup turn detection logging for an AgentSession.
    
    This is how servers would integrate the new turn detection logging feature.
    """
    
    # Create conversation tracker
    tracker = ConversationTracker()
    
    # Register event handlers
    @session.on("metrics_collected")
    def on_metrics_collected(event):
        """Handle all metrics events, filtering for turn detection."""
        tracker.handle_turn_detection(event)
    
    @session.on("speech_created")
    def on_speech_created(event):
        """Handle speech creation events for correlation."""
        tracker.handle_speech_created(event)
    
    # Optional: Set up periodic analytics reporting
    async def report_analytics():
        while True:
            await asyncio.sleep(60)  # Report every minute
            analytics = tracker.get_conversation_analytics()
            if analytics:
                logger.info("CONVERSATION_ANALYTICS", extra=analytics)
    
    # Start analytics task
    asyncio.create_task(report_analytics())
    
    return tracker


def main():
    """
    Demonstrate the turn detection logging setup.
    
    In a real application, this would be part of your agent server setup.
    """
    
    print("🎯 Turn Detection Logging Example")
    print("=" * 50)
    
    print("\n1. Server Setup:")
    print("   - Create AgentSession with turn detection enabled")
    print("   - Register metrics_collected and speech_created event handlers")
    print("   - Set up ConversationTracker for correlation")
    
    print("\n2. Expected Log Output:")
    print("   2024-01-15 10:30:15 - TURN_DETECTION: probability=0.850, turn_ended=True, inference=45ms, delay=2.0s, collision=1.0x")
    print("   2024-01-15 10:30:16 - CONVERSATION_TURN: turn=0.850, ended=True (45ms) → speech='Hi! How can I help?' latency=1.2s")
    
    print("\n3. Analytics Output:")
    print("   2024-01-15 10:31:00 - CONVERSATION_ANALYTICS: avg_probability=0.78, avg_inference=42ms, high_collisions=2")
    
    print("\n4. Integration Code:")
    print("""
    # In your server code:
    session = AgentSession(
        # ... other config ...
        turn_detection=your_turn_detector,  # Must have turn detection enabled
    )
    
    # Set up logging
    tracker = setup_turn_detection_logging(session)
    
    # Start agent
    await session.start(your_agent)
    """)
    
    print("\n✅ Turn detection logging is now ready for production use!")
    print("\nKey Benefits:")
    print("  • Complete visibility into turn detection decisions")
    print("  • Correlation between turn detection and speech generation")
    print("  • Performance monitoring (inference time, collision patterns)")
    print("  • Rich analytics for conversation optimization")
    print("  • Zero impact on existing functionality")


if __name__ == "__main__":
    main()