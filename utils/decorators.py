import functools
import time
import logging
from typing import Any, Callable, Optional
import hashlib
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(
            f"{func.__name__} executed in {execution_time:.4f} seconds")

        return result

    return wrapper


def log_decorator(level: str = 'INFO') -> Callable:
    """Decorator to log function calls and results"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            log_level = getattr(logging, level.upper())
            logger.log(log_level, f"Calling {func.__name__}")

            # Log arguments (be careful with sensitive data)
            if args:
                # Only first 3 args
                logger.log(log_level, f"  Args: {args[:3]}...")
            if kwargs:
                logger.log(log_level, f"  Kwargs: {list(kwargs.keys())}")

            try:
                result = func(*args, **kwargs)
                logger.log(
                    log_level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}")
                raise

        return wrapper

    return decorator


def retry_decorator(max_attempts: int = 3, delay: float = 1.0,
                    backoff: float = 2.0, exceptions: tuple = (Exception,)) -> Callable:
    """Decorator to retry function on failure"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts")
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {current_delay}s... Error: {str(e)}"
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

        return wrapper

    return decorator


def cache_decorator(ttl: Optional[int] = None, max_size: int = 128) -> Callable:
    """Decorator to cache function results"""

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = _make_cache_key(func.__name__, args, kwargs)

            # Check if cached result exists and is still valid
            if cache_key in cache:
                if ttl is None or (
                    cache_key in cache_times and
                    (datetime.now() - cache_times[cache_key]).seconds < ttl
                ):
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)

            # Implement simple LRU: remove oldest if cache is full
            if len(cache) >= max_size:
                oldest_key = min(cache_times.keys(),
                                 key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]

            cache[cache_key] = result
            cache_times[cache_key] = datetime.now()
            logger.debug(f"Cached result for {func.__name__}")

            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'max_size': max_size,
            'ttl': ttl
        }

        return wrapper

    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a hash key for caching"""
    try:
        # Convert args and kwargs to string representation
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    except Exception:
        # Fallback to string representation
        return str(hash((func_name, args, tuple(sorted(kwargs.items())))))


def validate_input(**validators) -> Callable:
    """Decorator to validate function inputs"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    if callable(validator):
                        if not validator(value):
                            raise ValueError(
                                f"Validation failed for parameter '{param_name}' "
                                f"with value: {value}"
                            )
                    elif isinstance(validator, type):
                        if not isinstance(value, validator):
                            raise TypeError(
                                f"Parameter '{param_name}' must be of type {validator.__name__}, "
                                f"got {type(value).__name__}"
                            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(reason: str = '', version: str = '') -> Callable:
    """Decorator to mark functions as deprecated"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__name__} is deprecated"
            if version:
                warning_msg += f" since version {version}"
            if reason:
                warning_msg += f". {reason}"

            logger.warning(warning_msg)
            import warnings
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_exceptions(*exception_types, default_return=None, log_error=True) -> Callable:
    """Decorator to handle specific exceptions gracefully"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    logger.error(
                        f"Exception in {func.__name__}: {type(e).__name__}: {str(e)}"
                    )
                return default_return

        return wrapper

    return decorator


def rate_limit(max_calls: int, time_window: int) -> Callable:
    """Decorator to limit function call rate"""

    def decorator(func: Callable) -> Callable:
        calls = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside time window
            nonlocal calls
            calls = [call_time for call_time in calls
                     if now - call_time < time_window]

            # Check rate limit
            if len(calls) >= max_calls:
                wait_time = time_window - (now - calls[0])
                raise RuntimeError(
                    f"Rate limit exceeded. Max {max_calls} calls per {time_window}s. "
                    f"Wait {wait_time:.1f}s"
                )

            # Record this call and execute
            calls.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_columns(*required_cols) -> Callable:
    """Decorator to validate DataFrame has required columns"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import pandas as pd

            # Find DataFrame in arguments
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break

            if df is None and 'df' in kwargs:
                df = kwargs['df']

            if df is None:
                raise ValueError("No DataFrame found in arguments")

            # Check for required columns
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"DataFrame missing required columns: {missing_cols}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def memoize_method(method: Callable) -> Callable:
    """Decorator to memoize class methods"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Create cache attribute on instance if it doesn't exist
        if not hasattr(self, '_method_cache'):
            self._method_cache = {}

        # Create cache key
        cache_key = (method.__name__, args, tuple(sorted(kwargs.items())))

        # Return cached result if available
        if cache_key in self._method_cache:
            return self._method_cache[cache_key]

        # Compute and cache result
        result = method(self, *args, **kwargs)
        self._method_cache[cache_key] = result

        return result

    return wrapper


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc

        # Start tracing
        tracemalloc.start()

        try:
            result = func(*args, **kwargs)

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()

            logger.info(
                f"{func.__name__} memory usage: "
                f"current={current / 1024 / 1024:.2f} MB, "
                f"peak={peak / 1024 / 1024:.2f} MB"
            )

            return result
        finally:
            tracemalloc.stop()

    return wrapper


def singleton_decorator(cls):
    """Decorator to make a class a singleton"""
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


# Decorator to measure pandas operations performance
def profile_pandas(func: Callable) -> Callable:
    """Decorator to profile pandas operations"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import pandas as pd

        # Record initial state
        start_time = time.time()
        start_memory = 0

        # Find DataFrame in arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                start_memory += arg.memory_usage(deep=True).sum()

        # Execute function
        result = func(*args, **kwargs)

        # Calculate metrics
        end_time = time.time()
        execution_time = end_time - start_time

        end_memory = 0
        if isinstance(result, pd.DataFrame):
            end_memory = result.memory_usage(deep=True).sum()

        memory_change = end_memory - start_memory

        logger.info(
            f"{func.__name__} pandas profile: "
            f"time={execution_time:.4f}s, "
            f"memory_change={memory_change / 1024 / 1024:.2f} MB"
        )

        return result

    return wrapper
