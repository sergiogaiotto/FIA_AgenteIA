"""
Rate limiting middleware
"""

import time
from collections import defaultdict
from typing import Dict, Callable
from fastapi import HTTPException, Request
from functools import wraps


class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, requests: int = 100, period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            requests: Number of requests allowed
            period: Time period in seconds
        """
        self.requests = requests
        self.period = period
        self.clients: Dict[str, list] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host if request.client else "unknown"
    
    def _is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make request"""
        now = time.time()
        
        # Clean old requests
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if req_time > now - self.period
        ]
        
        # Check rate limit
        if len(self.clients[client_id]) >= self.requests:
            return False
        
        # Add current request
        self.clients[client_id].append(now)
        return True
    
    def limit(self, func: Callable) -> Callable:
        """Decorator for rate limiting"""
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_id = self._get_client_id(request)
            
            if not self._is_allowed(client_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper