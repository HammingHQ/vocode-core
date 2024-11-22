import asyncio
from typing import Any, Awaitable, Callable, Dict

from loguru import logger


class HMEEventsManager:
    def __init__(self):
        self._handlers: Dict[str, list[Callable[..., Awaitable[Any]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event: str, handler: Callable[..., Awaitable[Any]]) -> None:
        async with self._lock:
            if event not in self._handlers:
                self._handlers[event] = []
            self._handlers[event].append(handler)

    async def emit(self, event: str, **kwargs) -> None:
        if handlers := self._handlers.get(event):
            for handler in handlers:
                try:
                    await handler(**kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")
