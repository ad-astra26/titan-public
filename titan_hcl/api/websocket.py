"""
api/websocket.py
Real-time WebSocket event stream for the Sovereign Observatory.

Connects browser clients to the EventBus for live updates on mood shifts,
social posts, epoch transitions, and directive changes.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket event stream.

    After connection, sends a JSON event per message:
    {
        "type": "mood_update" | "social_post" | "epoch_transition" | "directive_update" | ...,
        "data": { ... },
        "timestamp": 1710000000.0
    }

    Client can send JSON with {"type": "ping"} to keep alive.
    Server responds with {"type": "pong"}.
    """
    await ws.accept()

    # Get event bus from app state
    event_bus = getattr(ws.app.state, "event_bus", None)
    if event_bus is None:
        await ws.send_json({"type": "error", "data": {"detail": "Event bus not initialized."}})
        await ws.close(code=1011)
        return

    # Subscribe to events
    queue = event_bus.subscribe()
    logger.info("[WebSocket] Client connected. Subscribers: %d", event_bus.subscriber_count)

    try:
        # Run two tasks: reading from client + pushing events
        await asyncio.gather(
            _send_events(ws, queue),
            _receive_messages(ws),
        )
    except WebSocketDisconnect:
        logger.debug("[WebSocket] Client disconnected normally.")
    except Exception as e:
        logger.debug("[WebSocket] Connection closed: %s", e)
    finally:
        event_bus.unsubscribe(queue)
        logger.info("[WebSocket] Client removed. Subscribers: %d", event_bus.subscriber_count)


async def _send_events(ws: WebSocket, queue: asyncio.Queue):
    """Push events from the queue to the WebSocket client."""
    while True:
        event = await queue.get()
        try:
            await ws.send_json(event)
        except Exception:
            return  # Connection closed


async def _receive_messages(ws: WebSocket):
    """Handle incoming messages from the client (ping/pong keep-alive)."""
    while True:
        try:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
        except (json.JSONDecodeError, KeyError):
            pass  # Ignore malformed client messages
        except WebSocketDisconnect:
            raise
        except Exception:
            return
