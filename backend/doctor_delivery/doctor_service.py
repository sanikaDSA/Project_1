import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("sehat_doctor_delivery")

class DoctorDeliveryService:
    """Delivers completed pre-screening summaries and real-time emergency alerts to clinical queue."""
    def __init__(self):
        self.emergency_feed: List[Dict[str, Any]] = []
        self.completed_queue: List[Dict[str, Any]] = []

    def push_emergency_alert(self, emergency_event: Dict[str, Any]):
        self.emergency_feed.insert(0, emergency_event)
        logger.warning(f"🚨 CLINICAL EMERGENCY PUSHED TO DOCTOR FEED: {emergency_event.get('session_id')}")

    def push_completed_summary(self, summary: Dict[str, Any]):
        self.completed_queue.insert(0, summary)
        logger.info(f"✅ CLINICAL SUMMARY PUSHED TO DOCTOR QUEUE: {summary.get('session_id')}")

    def get_emergency_feed(self) -> List[Dict[str, Any]]:
        return self.emergency_feed

    def get_doctor_queue(self) -> List[Dict[str, Any]]:
        return self.completed_queue


# Global instance
GLOBAL_DOCTOR_DELIVERY = DoctorDeliveryService()
