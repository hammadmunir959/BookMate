"""
Step tracking system for document ingestion pipeline
Provides real-time progress updates and comprehensive reporting
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of an ingestion step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepInfo:
    """Information about a single ingestion step"""
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Mark step as started"""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = time.time()
        
    def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark step as completed"""
        self.status = StepStatus.COMPLETED
        self.end_time = time.time()
        self.duration = self.end_time - (self.start_time or self.end_time)
        if metadata:
            self.metadata.update(metadata)
            
    def fail(self, error_message: str):
        """Mark step as failed"""
        self.status = StepStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - (self.start_time or self.end_time)
        self.error_message = error_message


@dataclass
class IngestionReport:
    """Comprehensive report of document ingestion process"""
    document_id: str
    filename: str
    status: str  # "completed", "failed", "in_progress"
    steps: List[StepInfo]
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    error_message: Optional[str] = None
    total_duration: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def completed_steps(self) -> int:
        """Number of completed steps"""
        return len([step for step in self.steps if step.status == StepStatus.COMPLETED])
    
    @property
    def total_steps(self) -> int:
        """Total number of steps"""
        return len(self.steps)
    
    @property
    def progress_percentage(self) -> float:
        """Progress as percentage (0-100)"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    @property
    def current_step(self) -> Optional[StepInfo]:
        """Currently active step"""
        for step in self.steps:
            if step.status == StepStatus.IN_PROGRESS:
                return step
        return None


class IngestionStepTracker:
    """Tracks document ingestion steps and emits progress events"""
    
    # Define the standard ingestion steps
    STANDARD_STEPS = [
        "validation",
        "text_extraction", 
        "chunking",
        "embedding_generation",
        "vector_storage",
        "metadata_caching",
        "summary_generation"
    ]
    
    def __init__(self, document_id: str, filename: str, event_callback: Optional[Callable] = None):
        """
        Initialize step tracker
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            event_callback: Optional callback function for progress events
        """
        self.document_id = document_id
        self.filename = filename
        self.event_callback = event_callback
        
        # Initialize steps
        self.steps = {
            step_name: StepInfo(name=step_name) 
            for step_name in self.STANDARD_STEPS
        }
        
        # Create report
        self.report = IngestionReport(
            document_id=document_id,
            filename=filename,
            status="in_progress",
            steps=list(self.steps.values())
        )
        
        logger.info(f"Initialized step tracker for document {document_id}")
    
    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> StepInfo:
        """
        Start tracking a step
        
        Args:
            step_name: Name of the step to start
            metadata: Optional metadata for the step
            
        Returns:
            StepInfo object for the step
        """
        if step_name not in self.steps:
            # Add custom step if not in standard steps
            self.steps[step_name] = StepInfo(name=step_name)
            self.report.steps = list(self.steps.values())
        
        step = self.steps[step_name]
        step.start()
        
        if metadata:
            step.metadata.update(metadata)
        
        # Emit event
        self._emit_event({
            "event": "step_started",
            "document_id": self.document_id,
            "step": step_name,
            "status": "in_progress",
            "progress": f"{self.report.completed_steps}/{self.report.total_steps}",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Started step '{step_name}' for document {self.document_id}")
        return step
    
    def complete_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> StepInfo:
        """
        Complete a step
        
        Args:
            step_name: Name of the step to complete
            metadata: Optional metadata for the step
            
        Returns:
            StepInfo object for the step
        """
        if step_name not in self.steps:
            logger.warning(f"Attempted to complete unknown step: {step_name}")
            return None
        
        step = self.steps[step_name]
        step.complete(metadata)
        
        # Emit event
        self._emit_event({
            "event": "step_completed",
            "document_id": self.document_id,
            "step": step_name,
            "status": "completed",
            "progress": f"{self.report.completed_steps}/{self.report.total_steps}",
            "duration": f"{step.duration:.1f}s" if step.duration else "0s",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Completed step '{step_name}' for document {self.document_id} in {step.duration:.1f}s")
        return step
    
    def fail_step(self, step_name: str, error_message: str) -> StepInfo:
        """
        Mark a step as failed
        
        Args:
            step_name: Name of the step that failed
            error_message: Error description
            
        Returns:
            StepInfo object for the step
        """
        if step_name not in self.steps:
            logger.warning(f"Attempted to fail unknown step: {step_name}")
            return None
        
        step = self.steps[step_name]
        step.fail(error_message)
        
        # Mark report as failed
        self.report.status = "failed"
        self.report.error_message = error_message
        
        # Emit event
        self._emit_event({
            "event": "step_failed",
            "document_id": self.document_id,
            "step": step_name,
            "status": "failed",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.error(f"Failed step '{step_name}' for document {self.document_id}: {error_message}")
        return step
    
    def complete_ingestion(self, summary: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Mark ingestion as completed
        
        Args:
            summary: Generated document summary
            metadata: Additional metadata for the report
        """
        self.report.status = "completed"
        self.report.summary = summary
        
        if metadata:
            self.report.metadata.update(metadata)
        
        # Calculate total duration
        if self.report.steps:
            start_times = [step.start_time for step in self.report.steps if step.start_time]
            end_times = [step.end_time for step in self.report.steps if step.end_time]
            
            if start_times and end_times:
                self.report.total_duration = max(end_times) - min(start_times)
        
        # Emit final event
        self._emit_event({
            "event": "ingestion_completed",
            "document_id": self.document_id,
            "status": "completed",
            "total_duration": f"{self.report.total_duration:.1f}s" if self.report.total_duration else "0s",
            "chunks_created": self.report.metadata.get("chunks_created", 0),
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Completed ingestion for document {self.document_id}")
    
    def fail_ingestion(self, error_message: str):
        """
        Mark ingestion as failed
        
        Args:
            error_message: Error description
        """
        self.report.status = "failed"
        self.report.error_message = error_message
        
        # Emit failure event
        self._emit_event({
            "event": "ingestion_failed",
            "document_id": self.document_id,
            "status": "failed",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.error(f"Failed ingestion for document {self.document_id}: {error_message}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information
        
        Returns:
            Dictionary with progress information
        """
        current_step = self.report.current_step
        
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "status": self.report.status,
            "progress": f"{self.report.completed_steps}/{self.report.total_steps}",
            "progress_percentage": self.report.progress_percentage,
            "current_step": current_step.name if current_step else None,
            "completed_steps": self.report.completed_steps,
            "total_steps": self.report.total_steps,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_report(self) -> IngestionReport:
        """
        Get the complete ingestion report
        
        Returns:
            IngestionReport object
        """
        return self.report
    
    def _emit_event(self, event_data: Dict[str, Any]):
        """
        Emit a progress event
        
        Args:
            event_data: Event data to emit
        """
        if self.event_callback:
            try:
                # Check if the callback is a coroutine function
                import asyncio
                import inspect
                
                if inspect.iscoroutinefunction(self.event_callback):
                    # Schedule the coroutine to run in the event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create a task to run the coroutine
                            asyncio.create_task(self.event_callback(event_data))
                        else:
                            # Run the coroutine directly if no loop is running
                            loop.run_until_complete(self.event_callback(event_data))
                    except RuntimeError:
                        # If we can't get the event loop, just log the warning
                        logger.warning("Could not schedule async event callback - no event loop available")
                else:
                    # Regular function call
                    self.event_callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {str(e)}")
        
        # Also log the event for debugging
        logger.debug(f"Emitted event: {event_data}")


# Global registry for active trackers
_active_trackers: Dict[str, IngestionStepTracker] = {}

# Global event callback
_event_callback: Optional[Callable] = None


def set_event_callback(callback: Callable):
    """Set the global event callback for progress events"""
    global _event_callback
    _event_callback = callback


def get_tracker(document_id: str) -> Optional[IngestionStepTracker]:
    """Get active tracker for a document"""
    return _active_trackers.get(document_id)


def register_tracker(tracker: IngestionStepTracker):
    """Register a tracker in the global registry"""
    _active_trackers[tracker.document_id] = tracker
    
    # Set the global event callback if available
    if _event_callback:
        tracker.event_callback = _event_callback


def unregister_tracker(document_id: str):
    """Unregister a tracker from the global registry"""
    if document_id in _active_trackers:
        del _active_trackers[document_id]


def get_all_trackers() -> Dict[str, IngestionStepTracker]:
    """Get all active trackers"""
    return _active_trackers.copy()
