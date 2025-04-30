import time
from collections import deque
from functools import reduce
from typing import Deque, Generic, Optional, Protocol, TypeVar

T = TypeVar("T")


class Buffer(Generic[T]):
    """
    A buffer that stores the last `max_size` elements for a maximum of `max_life` seconds.
    Each element is stored with a timestamp, and elements older than `max_life` are automatically removed from the buffer.
    """

    def __init__(self, max_size: int, max_life: float = 1) -> None:
        assert max_size > 0
        assert max_life > 0

        self.max_size = max_size
        self.max_life = max_life
        self.buffer: Deque[T] = deque(maxlen=max_size)
        self.buffer_timestamps: Deque[float] = deque(maxlen=max_size)

    def add(self, value: T) -> None:
        """
        Add a value to the buffer.
        """
        self.buffer.append(value)
        self.buffer_timestamps.append(time.time())

    def _items(self) -> Deque[T]:
        while (
            len(self.buffer) > 0
            and time.time() - self.buffer_timestamps[0] > self.max_life
        ):
            self.buffer.popleft()
            self.buffer_timestamps.popleft()

        return self.buffer

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.buffer = deque(maxlen=self.max_size)
        self.buffer_timestamps = deque(maxlen=self.max_size)

    def mode(self) -> Optional[T]:
        """
        Return the most common element in the buffer.
        """
        items = self._items()

        if len(items) == 0:
            return None
        return max(set(items), key=items.count)

    def first(self) -> Optional[T]:
        """
        Return the first element in the buffer.
        """
        items = self._items()

        if len(items) == 0:
            return None
        return items[0]

    def last(self) -> Optional[T]:
        """
        Return the last element in the buffer.
        """
        items = self._items()

        if len(items) == 0:
            return None
        return items[-1]

    def __str__(self) -> str:
        return str(self._items())

    def __repr__(self) -> str:
        return str(self)


U = TypeVar("U", bound="UBound")


class UBound(Protocol):
    def __add__(self: T, other: T) -> T: ...
    def __truediv__(self: T, other: int) -> T: ...


class ArithmeticBuffer(Buffer[U]):
    """
    A buffer that stores the last `max_size` elements for a maximum of `max_life` seconds.
    Each element is stored with a timestamp, and elements older than `max_life` are automatically removed from the buffer.
    The buffer supports arithmetic operations on the elements.
    """

    def __init__(self, max_size: int, max_life: float = 1) -> None:
        super().__init__(max_size, max_life)

    def average(self) -> Optional[U]:
        """
        Return the average of the elements in the buffer.
        """
        items = self._items()

        if len(items) == 0:
            return None
        return reduce(lambda x, y: x + y, items) / len(items)
