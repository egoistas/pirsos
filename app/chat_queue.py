"""FIFO admission for the single API worker configured in Dockerfile.

Slots represent requests, not authenticated people. All mutations run on the
same event loop; do not deploy multiple API workers with this in-memory queue.
"""
from collections import deque


class ChatQueue:
    def __init__(self):
        self.waiting = deque()
        self.active = set()
        self.limit = 2

    def join(self):
        ticket = object()
        self.waiting.append(ticket)
        return ticket

    def claim(self, ticket):
        if len(self.active) < self.limit and self.waiting and self.waiting[0] is ticket:
            self.waiting.popleft()
            self.active.add(ticket)
            return True
        return False

    def position(self, ticket):
        return list(self.waiting).index(ticket) + 1

    def release(self, ticket):
        self.active.discard(ticket)
        if ticket in self.waiting:
            self.waiting.remove(ticket)


chat_queue = ChatQueue()
