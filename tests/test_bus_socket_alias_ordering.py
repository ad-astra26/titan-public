"""Regression guard for BUG-BROKER-ORPHAN-SUB-WARN-FROM-ALIAS-REGISTRATION-20260526.

Root cause: `subscribe_alias()` was called synchronously right after the
async `client.start()`, so it buffered its alias BUS_SUBSCRIBE frame into the
FIFO `_outbound_buffer` BEFORE the connection thread sent the primary
subscribe frame. On connect the buffer flushed alias-first → the Rust broker
promoted the alias name (`meta_service` / `guardian_hcl_lifecycle`) to PRIMARY
with empty topics + reply_only=false → it WARN-dropped every dst="all"
broadcast in the window before the primary subscribe landed.

Fix: `subscribe_alias()` defers the immediate frame send until the primary
subscribe has gone out on the current connection (`_primary_subscribed`). The
alias is still registered in `self._aliases`, so the connect sequence fires
the primary FIRST then one alias frame per entry — guaranteeing
alias-after-primary ordering. These tests pin that gate.
"""

from __future__ import annotations

from titan_hcl.core.bus_socket import BusSocketClient


def _client(tmp_path) -> BusSocketClient:
    # Construct only — never .start(); we exercise the pre-connect gate, so no
    # broker/socket is needed. _primary_subscribed starts unset (boot state).
    return BusSocketClient(
        titan_id="testT",
        authkey=b"k" * 32,
        name="cognitive_worker",
        sock_path=tmp_path / "bus.sock",
        topics=["SPIRIT_STATE"],
    )


def test_alias_before_primary_subscribe_is_deferred_not_buffered(tmp_path):
    """At boot (primary subscribe not yet sent) subscribe_alias must register
    the alias but NOT buffer a frame — otherwise it races ahead of the primary
    subscribe in the FIFO outbound buffer and the broker promotes it to an
    empty-topics primary."""
    c = _client(tmp_path)
    assert not c._primary_subscribed.is_set(), "fresh client must start with the gate closed"
    assert len(c._outbound_buffer) == 0

    c.subscribe_alias("meta_service")

    # Registered for the connect sequence to send (primary-first)...
    assert "meta_service" in c._aliases
    # ...but NOT sent now (would land ahead of the primary subscribe).
    assert len(c._outbound_buffer) == 0, (
        "alias frame was buffered before the primary subscribe — this is the "
        "exact ordering that makes the broker promote the alias to primary")


def test_alias_after_primary_subscribe_sends_immediately(tmp_path):
    """A runtime alias add (after the primary subscribe is on the wire) must
    send immediately — the connect sequence won't re-run until the next
    reconnect, so deferring would drop the dst-route until then."""
    c = _client(tmp_path)
    c._primary_subscribed.set()  # simulate: primary subscribe already sent

    c.subscribe_alias("meta_service")

    assert "meta_service" in c._aliases
    assert len(c._outbound_buffer) == 1, "runtime alias add must be sent immediately"


def test_disconnect_rearms_the_gate(tmp_path):
    """The connect/disconnect lifecycle must re-arm the gate so the NEXT
    connection again sends the primary subscribe before alias frames."""
    c = _client(tmp_path)
    c._primary_subscribed.set()
    # Simulate the connection_loop finally-block clearing on disconnect.
    c._primary_subscribed.clear()
    assert not c._primary_subscribed.is_set()

    c.subscribe_alias("meta_service")
    assert "meta_service" in c._aliases
    assert len(c._outbound_buffer) == 0, "post-disconnect alias add must defer again"
