"""
Unified Cache Manager for RAG Microservice
Handles caching for embeddings, responses, and temporary data
"""

import logging
import time
import hashlib
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from src.core.config import config

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache manager for all caching needs"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._ensure_cache_directories()

    def _ensure_cache_directories(self):
        """Ensure cache directories exist"""
        for cache_path in [config.cache.embedding_cache_path,
                          config.cache.response_cache_path,
                          config.cache.temp_cache_path]:
            Path(cache_path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            # Check memory cache first
            if key in self.cache:
                cached_item = self.cache[key]
                if time.time() - cached_item['timestamp'] < (cached_item['ttl_hours'] * 3600):
                    return cached_item['value']
                else:
                    # Expired, remove from cache
                    del self.cache[key]

            # Check file cache
            return self._get_from_file_cache(key)

        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl_hours: int = 24) -> bool:
        """Set cached value"""
        try:
            # Store in memory cache
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl_hours': ttl_hours
            }

            # Store in file cache for persistence
            self._set_to_file_cache(key, value, ttl_hours)

            return True

        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {str(e)}")
            return False

    def _get_cache_path(self, key: str, cache_type: str = "responses") -> Path:
        """Get cache file path"""
        # Create a safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_dir = getattr(config.cache, f"{cache_type}_cache_path")
        return Path(cache_dir) / f"{safe_key}.json"

    def _get_from_file_cache(self, key: str, cache_type: str = "responses") -> Optional[Any]:
        """Get value from file cache"""
        try:
            cache_file = self._get_cache_path(key, cache_type)
            if not cache_file.exists():
                return None

            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if expired
            if time.time() - cached_data['timestamp'] > (cached_data['ttl_hours'] * 3600):
                # Remove expired file
                cache_file.unlink()
                return None

            return cached_data['value']

        except Exception as e:
            logger.warning(f"File cache get failed for key {key}: {str(e)}")
            return None

    def _set_to_file_cache(self, key: str, value: Any, ttl_hours: int, cache_type: str = "responses") -> bool:
        """Set value to file cache"""
        try:
            cache_file = self._get_cache_path(key, cache_type)

            # Ensure directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            cached_data = {
                'key': key,
                'value': value,
                'timestamp': time.time(),
                'ttl_hours': ttl_hours
            }

            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, default=str)

            return True

        except Exception as e:
            logger.warning(f"File cache set failed for key {key}: {str(e)}")
            return False

    def delete(self, key: str, cache_type: str = "responses") -> bool:
        """Delete cached value"""
        try:
            # Remove from memory cache
            if key in self.cache:
                del self.cache[key]

            # Remove from file cache
            cache_file = self._get_cache_path(key, cache_type)
            if cache_file.exists():
                cache_file.unlink()

            return True

        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {str(e)}")
            return False

    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        try:
            cleared_count = 0

            # Clear memory cache
            current_time = time.time()
            expired_keys = []
            for key, cached_item in self.cache.items():
                if current_time - cached_item['timestamp'] > (cached_item['ttl_hours'] * 3600):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                cleared_count += 1

            # Clear file caches
            for cache_type in ["embedding", "response", "temp"]:
                cache_dir = getattr(config.cache, f"{cache_type}_cache_path")
                if os.path.exists(cache_dir):
                    for cache_file in Path(cache_dir).glob("*.json"):
                        try:
                            with open(cache_file, 'r') as f:
                                cached_data = json.load(f)

                            if current_time - cached_data['timestamp'] > (cached_data['ttl_hours'] * 3600):
                                cache_file.unlink()
                                cleared_count += 1
                        except Exception:
                            # Remove corrupted cache files
                            cache_file.unlink()
                            cleared_count += 1

            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count

        except Exception as e:
            logger.warning(f"Clear expired cache failed: {str(e)}")
            return 0

    def clear_all(self) -> int:
        """Clear all cache entries"""
        try:
            cleared_count = len(self.cache)
            self.cache.clear()

            # Clear file caches
            for cache_type in ["embedding", "response", "temp"]:
                cache_dir = getattr(config.cache, f"{cache_type}_cache_path")
                if os.path.exists(cache_dir):
                    for cache_file in Path(cache_dir).glob("*.json"):
                        cache_file.unlink()
                        cleared_count += 1

            logger.info(f"Cleared all {cleared_count} cache entries")
            return cleared_count

        except Exception as e:
            logger.warning(f"Clear all cache failed: {str(e)}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                'memory_cache_size': len(self.cache),
                'file_cache_sizes': {}
            }

            # Count files in each cache directory
            for cache_type in ["embedding", "response", "temp"]:
                cache_dir = getattr(config.cache, f"{cache_type}_cache_path")
                if os.path.exists(cache_dir):
                    file_count = len(list(Path(cache_dir).glob("*.json")))
                    stats['file_cache_sizes'][cache_type] = file_count
                else:
                    stats['file_cache_sizes'][cache_type] = 0

            return stats

        except Exception as e:
            logger.warning(f"Get cache stats failed: {str(e)}")
            return {'error': str(e)}
