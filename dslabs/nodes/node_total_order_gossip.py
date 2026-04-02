from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set, Callable
import heapq
import uuid
import random


@dataclass
class NodeTotalOrderGossip:
    """
    Gossip-based total order broadcast. Each node gossips received messages
    to a subset of peers over multiple rounds. Uses Lamport clock + node id
    to induce a total order on events. Use a cache to deduplicate messages.
    """

    node_id: str
    peers: List[str]
    transport: Any
    scheduler: Any
    num_gossip_peers: int
    num_gossip_rounds: int
    # Key-value store
    store: Dict[str, Any] = field(default_factory=dict)
    # Append-only log of all messages delivered
    log: List[Tuple[str, str, Any]] = field(default_factory=list)
    # Remember which messages we have already received
    cache: Set[str] = field(default_factory=set)
    # Lamport clock for global message ordering
    lamport_clock: int = field(default_factory=int)
    # Buffer for delayed messages with a default timeout of 1s
    buffer: List[Tuple[int, str, str, str, Any, Callable[[], None]]] = field(
        default_factory=list
    )
    buffer_timeout: int = field(default_factory=lambda: 1000)
    verbose: bool = field(default_factory=lambda: False)

    def brief_state(self):
        return {"id": self.node_id, "kv": dict(self.store)}

    # Client-facing APIs (direct calls from the demo)
    def client_put(self, key, value):
        # Generate a unique id for this message (so we can dedup later)
        uid = str(uuid.uuid4())
        # Generate message node id and logical clock time
        self.lamport_clock += 1
        msg_clock = self.lamport_clock
        msg_node_id = self.node_id
        self.receive_and_replicate(uid, msg_clock, msg_node_id, key, value)

    def client_get(self, key):
        return self.store.get(key)

    # Node internal handlers
    def receive_and_replicate(self, uid, msg_clock, msg_node_id, key, value):
        if uid not in self.cache:
            self.receive(uid, msg_clock, msg_node_id, key, value)
            # Start gossip rounds instead of broadcasting to all peers
            self.start_gossip_rounds(uid, msg_clock, msg_node_id, key, value, round_num=0)

    def start_gossip_rounds(self, uid, msg_clock, msg_node_id, key, value, round_num):
        """Start gossip rounds for this message."""
        if round_num >= self.num_gossip_rounds:
            return  # All rounds completed

        # Select random peers for this round
        available_peers = [p for p in self.peers if p != self.node_id]
        if not available_peers:
            return

        # Select up to num_gossip_peers random peers
        selected_peers = random.sample(available_peers, min(self.num_gossip_peers, len(available_peers)))

        # Send to selected peers
        self.lamport_clock += 1
        for peer in selected_peers:
            self.transport.send(
                peer,
                {
                    "type": "replicate",
                    "from": self.node_id,
                    "msg_node_id": msg_node_id,
                    "msg_clock": msg_clock,
                    "uid": uid,
                    "key": key,
                    "value": value,
                },
            )

        # Schedule next round after 100ms
        def next_round():
            self.start_gossip_rounds(uid, msg_clock, msg_node_id, key, value, round_num + 1)

        self.scheduler.call_later(100, next_round)

    def receive(self, uid, msg_clock, msg_node_id, key, value):
        if self.verbose:
            print(f"NODE {self.node_id} RECV:", uid, msg_clock, msg_node_id, key, value)
        self.cache.add(uid)

        # Received messages are buffered to preserve total order
        def _deliver():
            self.deliver_up_to(uid)

        cancel = self.scheduler.call_later(self.buffer_timeout, _deliver)
        heapq.heappush(self.buffer, (msg_clock, msg_node_id, uid, key, value, cancel))

    def deliver_up_to(self, final_uid):
        """
        Deliver all messages in buffer order up to the given message uid.
         1. Append the delivery to the permanent log
         2. Update the key-value store
        """
        while True:
            msg_clock, msg_node_id, uid, key, value, cancel = heapq.heappop(self.buffer)
            self.log.append((uid, key, value))
            self.store[key] = value
            if uid == final_uid:
                break
            else:
                cancel()

    # Network handler
    def on_message(self, msg):
        typ = msg.get("type")
        if typ == "replicate":
            # Apply incoming replication from any peer.
            self.lamport_clock = max(self.lamport_clock, msg["msg_clock"]) + 1
            self.receive_and_replicate(
                msg["uid"],
                msg["msg_clock"],
                msg["msg_node_id"],
                msg["key"],
                msg["value"],
            )
        else:
            raise ValueError(f"Unknown type in message {msg!r}")