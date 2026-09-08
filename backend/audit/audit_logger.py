import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("sehat_audit")

class AuditLogger:
    """
    Structured Audit Logger for Sehat AI clinical events.
    Every event contains session_id, timestamp, event_type, and event_data.
    """
    def __init__(self, log_file: Optional[str] = "outputs/logs/sehat_audit.jsonl"):
        self.log_file = log_file
        self.in_memory_events: List[Dict[str, Any]] = []

    def log_event(self, session_id: str, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        event_record = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        self.in_memory_events.append(event_record)
        logger.info(f"AUDIT [{event_type}] Session: {session_id} | Data: {json.dumps(event_data, ensure_ascii=False)}")
        
        if self.log_file:
            try:
                import os
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event_record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to append to audit log file: {e}")
                
        return event_record

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.in_memory_events if e["session_id"] == session_id]

# Global instance
GLOBAL_AUDIT_LOGGER = AuditLogger()
