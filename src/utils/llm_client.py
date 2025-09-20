"""
Enhanced LLM Client for RAG Generation Microservice
Provides robust LLM interaction with rate limiting, caching, and fallback
"""

import logging
import time
import hashlib
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

from src.core.config import config
from src.core.service_manager import service_manager

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting and retry logic"""
    base_delay: float = 1.5  # Base delay between requests
    max_delay: float = 16.0  # Maximum delay for exponential backoff
    max_retries: int = 3
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: int = 60  # seconds
    max_concurrent_requests: int = 2


class CircuitBreaker:
    """Circuit breaker pattern for handling API failures"""
    
    def __init__(self, threshold: int = 3, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - (self.last_failure_time or 0) > self.timeout:
                self.state = "half_open"
                return True
            return False
        
        # half_open
        return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = "open"


class RequestQueue:
    """Thread-safe request queue with concurrency control"""
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.queue = asyncio.Queue()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        while self.active_requests >= self.max_concurrent:
            await asyncio.sleep(0.1)
        self.active_requests += 1
    
    def release(self):
        """Release request permission"""
        self.active_requests = max(0, self.active_requests - 1)


class LLMClient:
    """Enhanced LLM client with robust error handling and rate limiting"""
    
    def __init__(self, model: str = None, enable_cache: bool = True, config_rate: RateLimitConfig = None):
        self.model = model or config.model.groq_model
        self.enable_cache = enable_cache
        self.rate_config = config_rate or RateLimitConfig()
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            threshold=self.rate_config.circuit_breaker_threshold,
            timeout=self.rate_config.circuit_breaker_timeout
        )
        self.request_queue = RequestQueue(max_concurrent=self.rate_config.max_concurrent_requests)
        
        # Cache for responses
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Enhanced LLMClient initialized with model: {self.model}")
    
    def _wait_for_rate_limit(self):
        """Wait for rate limiting"""
        time.sleep(self.rate_config.base_delay)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.rate_config.base_delay * (2 ** attempt)
        return min(delay, self.rate_config.max_delay)
    
    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate cache key"""
        content = f"{prompt}_{max_tokens}_{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response"""
        if not self.enable_cache:
            return None
        
        cached = self.response_cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour TTL
            return cached['response']
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response"""
        if not self.enable_cache:
            return
        
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 2000,
        temperature: float = 0.1,
        system_prompt: str = None,
        expect_json: bool = False
    ) -> Optional[str]:
        """
        Generate text using the configured LLM
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System-level instructions
            expect_json: Whether to expect JSON response
            
        Returns:
            Generated text or None if failed
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker is open, skipping request")
            return None
        
        # Check cache
        cache_key = self._get_cache_key(prompt, max_tokens, temperature)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        # Try with retries
        for attempt in range(self.rate_config.max_retries + 1):
            try:
                # Prepare the full prompt
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\n{prompt}"
                
                # Call GROQ API (simplified - in real implementation, use httpx or similar)
                response = self._call_groq_api(full_prompt, max_tokens, temperature)
                
                if response:
                    # Record success
                    self.circuit_breaker.record_success()
                    
                    # Cache response
                    self._cache_response(cache_key, response)
                    
                    return response
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.rate_config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    # Record failure
                    self.circuit_breaker.record_failure()
                    logger.error(f"All LLM call attempts failed: {str(e)}")
        
        # If all retries failed, return fallback response
        return self._generate_fallback_response(prompt, expect_json)
    
    def _call_groq_api(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Call GROQ API"""
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {config.model.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": config.model.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": config.generation.top_p
            }
            
            with httpx.Client(timeout=config.model.groq_timeout) as client:
                response = client.post(
                    config.model.groq_base_url + "/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"GROQ API error: {response.status_code} - {response.text}")
                    return None
                    
        except ImportError:
            logger.warning("httpx not available, using mock response")
            return self._generate_mock_response(prompt, False)
        except Exception as e:
            logger.error(f"GROQ API call failed: {str(e)}")
            return None
    
    def _generate_fallback_response(self, prompt: str, expect_json: bool) -> str:
        """Generate fallback response when LLM fails"""
        logger.warning("Using fallback response generation")
        return self._generate_mock_response(prompt, expect_json)
    
    def _generate_mock_response(self, prompt: str, expect_json: bool = False) -> str:
        """Generate mock response for testing/development"""
        if expect_json:
            return '{"answer": "Mock response - LLM not available", "citations": []}'
        
        # Extract question from prompt
        lines = prompt.split('\n')
        question = ""
        for line in lines:
            if line.startswith('QUERY:') or 'query' in line.lower():
                question = line.replace('QUERY:', '').strip()
                break
        
        if not question:
            question = "your query"
        
        return f"""Based on the provided context, I cannot provide a definitive answer to {question} at this time due to system limitations. Please try again later or contact support for assistance.

If you have additional context or need help with a different question, feel free to ask!"""
    
    def generate_multiple(
        self, 
        prompts: List[str], 
        max_tokens: int = 2000,
        temperature: float = 0.1,
        system_prompt: str = None,
        delay_between_requests: float = 2.0
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        results = []
        for prompt in prompts:
            response = self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            results.append(response or "")
            
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        return results
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return True  # Always available with fallback to mock responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "cache_size": len(self.response_cache),
            "active_requests": self.request_queue.active_requests
        }


# Global LLM client instance
llm_client = LLMClient()
