"""
Error handling utilities
"""

import logging
from typing import Callable, Any
from functools import wraps
from fastapi import HTTPException


class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle(self, func: Callable) -> Callable:
        """Decorator for error handling"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except ValueError as e:
                self.logger.warning(f"Value error in {func.__name__}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except PermissionError as e:
                self.logger.warning(f"Permission error in {func.__name__}: {e}")
                raise HTTPException(status_code=403, detail="Permission denied")
            except FileNotFoundError as e:
                self.logger.warning(f"File not found in {func.__name__}: {e}")
                raise HTTPException(status_code=404, detail="Resource not found")
            except TimeoutError as e:
                self.logger.error(f"Timeout in {func.__name__}: {e}")
                raise HTTPException(status_code=504, detail="Request timeout")
            except Exception as e:
                self.logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error"
                )
        
        return wrapper