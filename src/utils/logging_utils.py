"""
Enhanced Logging Utilities for RAG Microservice
Provides structured logging, performance monitoring, and consistent logging patterns
"""

import logging
import time
import functools
from typing import Any, Dict, Optional, Callable, List
from contextlib import contextmanager
import asyncio
import json
from datetime import datetime

from src.core.config import config


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name

    def log_operation(self, operation: str, duration: float, **kwargs):
        """Log operation completion with timing"""
        if config.logging.enable_performance_logging:
            if duration > (config.logging.log_slow_operations_threshold_ms / 1000):
                level = "WARNING"
                message = f"SLOW OPERATION: {operation} took {duration:.3f}s"
            else:
                level = "DEBUG"
                message = f"OPERATION: {operation} completed in {duration:.3f}s"

            self._log_structured(level, message, operation=operation, duration=duration, **kwargs)

    def log_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Log API request details"""
        level = "WARNING" if status_code >= 400 else "INFO"
        message = f"API {method} {endpoint} -> {status_code} ({duration:.3f}s)"

        self._log_structured(level, message,
                           method=method,
                           endpoint=endpoint,
                           status_code=status_code,
                           duration=duration)

    def log_retrieval(self, query: str, results_count: int, duration: float, **kwargs):
        """Log retrieval operation details"""
        message = f"RETRIEVAL: '{query[:50]}...' -> {results_count} results ({duration:.3f}s)"
        self._log_structured("INFO", message,
                           query_length=len(query),
                           results_count=results_count,
                           duration=duration,
                           **kwargs)

    def log_generation(self, prompt_length: int, response_length: int, duration: float, **kwargs):
        """Log generation operation details"""
        message = f"GENERATION: {prompt_length} chars -> {response_length} chars ({duration:.3f}s)"
        self._log_structured("INFO", message,
                           prompt_length=prompt_length,
                           response_length=response_length,
                           duration=duration,
                           **kwargs)

    def log_ingestion_step(self, step_name: str, document_id: str, status: str, duration: float = None, **kwargs):
        """Log detailed ingestion pipeline steps"""
        if duration is not None:
            message = f"📥 INGESTION | {document_id[:8]} | {step_name} | {status} ({duration:.3f}s)"
        else:
            message = f"📥 INGESTION | {document_id[:8]} | {step_name} | {status}"

        level = "ERROR" if "failed" in status.lower() or "error" in status.lower() else "INFO"
        self._log_structured(level, message, step=step_name, document_id=document_id, **kwargs)

    def log_retrieval_step(self, step_name: str, query_id: str, status: str, duration: float = None, **kwargs):
        """Log detailed retrieval pipeline steps"""
        if duration is not None:
            message = f"🔍 RETRIEVAL | {query_id[:8]} | {step_name} | {status} ({duration:.3f}s)"
        else:
            message = f"🔍 RETRIEVAL | {query_id[:8]} | {step_name} | {status}"

        level = "ERROR" if "failed" in status.lower() or "error" in status.lower() else "INFO"
        self._log_structured(level, message, step=step_name, query_id=query_id, **kwargs)

    def log_generation_step(self, step_name: str, request_id: str, status: str, duration: float = None, **kwargs):
        """Log detailed generation pipeline steps"""
        if duration is not None:
            message = f"🤖 GENERATION | {request_id[:8]} | {step_name} | {status} ({duration:.3f}s)"
        else:
            message = f"🤖 GENERATION | {request_id[:8]} | {step_name} | {status}"

        level = "ERROR" if "failed" in status.lower() or "error" in status.lower() else "INFO"
        self._log_structured(level, message, step=step_name, request_id=request_id, **kwargs)

    def log_pipeline_start(self, pipeline_type: str, pipeline_id: str, **kwargs):
        """Log pipeline start"""
        message = f"▶️  {pipeline_type.upper()} | {pipeline_id[:8]} | PIPELINE START"
        self._log_structured("INFO", message, pipeline_type=pipeline_type, pipeline_id=pipeline_id, **kwargs)

    def log_pipeline_end(self, pipeline_type: str, pipeline_id: str, duration: float, status: str = "completed", **kwargs):
        """Log pipeline completion"""
        message = f"⏹️  {pipeline_type.upper()} | {pipeline_id[:8]} | PIPELINE END | {status} ({duration:.3f}s)"
        level = "ERROR" if "failed" in status.lower() else "INFO"
        self._log_structured(level, message, pipeline_type=pipeline_type, pipeline_id=pipeline_id, duration=duration, **kwargs)

    def _log_structured(self, level: str, message: str, **kwargs):
        """Internal method for structured logging"""
        try:
            if config.logging.enable_structured_logging:
                # Create JSON log entry
                log_entry = {
                    "level": level,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                full_message = json.dumps(log_entry)
            else:
                extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
                if extra_info:
                    full_message = f"{message} | {extra_info}"
                else:
                    full_message = message


            self.logger.log(getattr(logging, level.upper(), logging.INFO), full_message)

        except Exception as e:
            self.logger.warning(f"Failed to log structured data: {str(e)}")
            self.logger.log(getattr(logging, level.upper(), logging.INFO), message)


def log_operation(operation_name: str = None):
    """Decorator to log function execution time and details"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_operation(operation_name or func.__name__, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_operation(operation_name or func.__name__, duration, success=False, error=str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_operation(operation_name or func.__name__, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_operation(operation_name or func.__name__, duration, success=False, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_pipeline_step(pipeline_type: str, step_name: str = None):
    """Decorator to log pipeline steps with timing"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            pipeline_id = kwargs.get('document_id') or kwargs.get('query_id') or kwargs.get('request_id') or 'unknown'
            step = step_name or func.__name__
            start_time = time.time()

            # Log step start
            logger.log_ingestion_step(step, pipeline_id, "started") if pipeline_type == "ingestion" else \
            logger.log_retrieval_step(step, pipeline_id, "started") if pipeline_type == "retrieval" else \
            logger.log_generation_step(step, pipeline_id, "started")

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Log step completion
                if pipeline_type == "ingestion":
                    logger.log_ingestion_step(step, pipeline_id, "completed", duration)
                elif pipeline_type == "retrieval":
                    logger.log_retrieval_step(step, pipeline_id, "completed", duration)
                else:
                    logger.log_generation_step(step, pipeline_id, "completed", duration)

                return result
            except Exception as e:
                duration = time.time() - start_time

                # Log step failure
                if pipeline_type == "ingestion":
                    logger.log_ingestion_step(step, pipeline_id, f"failed: {str(e)}", duration)
                elif pipeline_type == "retrieval":
                    logger.log_retrieval_step(step, pipeline_id, f"failed: {str(e)}", duration)
                else:
                    logger.log_generation_step(step, pipeline_id, f"failed: {str(e)}", duration)

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            pipeline_id = kwargs.get('document_id') or kwargs.get('query_id') or kwargs.get('request_id') or 'unknown'
            step = step_name or func.__name__
            start_time = time.time()

            # Log step start
            logger.log_ingestion_step(step, pipeline_id, "started") if pipeline_type == "ingestion" else \
            logger.log_retrieval_step(step, pipeline_id, "started") if pipeline_type == "retrieval" else \
            logger.log_generation_step(step, pipeline_id, "started")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log step completion
                if pipeline_type == "ingestion":
                    logger.log_ingestion_step(step, pipeline_id, "completed", duration)
                elif pipeline_type == "retrieval":
                    logger.log_retrieval_step(step, pipeline_id, "completed", duration)
                else:
                    logger.log_generation_step(step, pipeline_id, "completed", duration)

                return result
            except Exception as e:
                duration = time.time() - start_time

                # Log step failure
                if pipeline_type == "ingestion":
                    logger.log_ingestion_step(step, pipeline_id, f"failed: {str(e)}", duration)
                elif pipeline_type == "retrieval":
                    logger.log_retrieval_step(step, pipeline_id, f"failed: {str(e)}", duration)
                else:
                    logger.log_generation_step(step, pipeline_id, f"failed: {str(e)}", duration)

                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def log_context(operation: str, **context_data):
    """Context manager for logging operation start/end with context"""
    logger = StructuredLogger(__name__)
    start_time = time.time()

    logger.logger.info(f"START: {operation}", extra=context_data if context_data else None)

    try:
        yield
        duration = time.time() - start_time
        logger.logger.info(f"END: {operation} (completed in {duration:.3f}s)", extra=context_data if context_data else None)
    except Exception as e:
        duration = time.time() - start_time
        logger.logger.error(f"END: {operation} (failed after {duration:.3f}s): {str(e)}", extra=context_data if context_data else None)
        raise


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


# Global logger instances for common use
api_logger = StructuredLogger("api")
retrieval_logger = StructuredLogger("retrieval")
generation_logger = StructuredLogger("generation")
ingestion_logger = StructuredLogger("ingestion")
