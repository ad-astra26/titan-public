"""
V3 Module Workers — supervised processes that communicate via the Divine Bus.

Each worker runs in its own process, started by the Guardian on demand.
Workers receive messages on their Bus queue and send responses back.
"""
