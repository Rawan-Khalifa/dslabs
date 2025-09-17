import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set


@dataclass
class NodeEagerBroadcast:
    """
    Eager broadcasting nodes. Every node sends every message it receives to every other node,
    including replication messages. Uses message deduplication to prevent infinite loops.
    """

    node_id: str
    peers: List[str]
    transport: Any
    scheduler: Any
    store: Dict[str, Any] = field(default_factory=dict)
    log: List[Tuple[str, Any]] = field(default_factory=list)
    seen_messages: Set[str] = field(default_factory=set)  # Track seen message IDs

    def brief_state(self):
        return {"id": self.node_id, "kv": dict(self.store)}

    # Client-facing APIs (direct calls from the demo)
    def client_put(self, key, value):
        # Create a unique message ID for this new message
        message_id = str(uuid.uuid4())
        # Accept locally, then replicate to others.
        self.receive_and_replicate(key, value, message_id)

    def client_get(self, key):
        return self.store.get(key)

    # Node internal handlers
    def receive_and_replicate(self, key, value, message_id):
        # Check if we've already seen this message
        if message_id in self.seen_messages:
            return  # Don't process duplicate messages
        
        # Mark this message as seen
        self.seen_messages.add(message_id)
        
        # Process the message locally
        self.receive(key, value)
        
        # Broadcast to all peers (eager broadcasting)
        for peer in self.peers:
            if peer != self.node_id:
                self.transport.send(
                    peer,
                    {
                        "type": "replicate",
                        "from": self.node_id,
                        "key": key,
                        "value": value,
                        "message_id": message_id,  # Include the message ID
                    },
                )

    def receive(self, key, value):
        # Received messages are delivered immediately
        self.deliver(key, value)

    def deliver(self, key, value):
        """
        Deliver the message to this node.
         1. Update the key-value store
         2. Append the delivery to the permanent log
        """
        self.store[key] = value
        self.log.append((key, value))

    # Network handler
    def on_message(self, msg):
        typ = msg.get("type")
        if typ == "replicate":
            # Apply incoming replication from any peer.
            # Use eager broadcasting - forward this message to all other peers too
            message_id = msg.get("message_id")
            if message_id:
                self.receive_and_replicate(msg["key"], msg["value"], message_id)
            else:
                # Fallback for messages without ID (shouldn't happen in normal operation)
                self.receive(msg["key"], msg["value"])
        else:
            raise ValueError(f"Unknown type in message {msg!r}")
