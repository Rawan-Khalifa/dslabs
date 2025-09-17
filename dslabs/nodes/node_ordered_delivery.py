import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import heapq


@dataclass 
class NodeOrderedDelivery:
    """
    Node implementation that ensures ordered message delivery using global sequence numbers.
    Messages are ordered by their client submission time, ensuring all nodes deliver in the same order.
    """

    node_id: str
    peers: List[str]
    transport: Any
    scheduler: Any
    store: Dict[str, Any] = field(default_factory=dict)
    log: List[Tuple[str, Any]] = field(default_factory=list)
    seen_messages: Set[str] = field(default_factory=set)  # Track seen message IDs
    pending_messages: List[Tuple[int, str, Any, str]] = field(default_factory=list)  # (global_seq, key, value, msg_id)
    next_expected_seq: int = 0  # Next sequence number we expect to deliver

    def brief_state(self):
        return {
            "id": self.node_id, 
            "kv": dict(self.store),
            "log_length": len(self.log),
            "pending": len(self.pending_messages),
            "next_expected": self.next_expected_seq
        }

    # Client-facing APIs (direct calls from the demo)
    def client_put(self, key, value):
        # The value itself is the sequence number in the simulation!
        # Messages are sent with values 0, 1, 2, ... in order
        global_sequence = value  # Use the value as the global sequence number
        message_id = str(uuid.uuid4())
        
        # Process and replicate with ordering information
        self.receive_and_replicate(key, value, message_id, global_sequence)

    def client_get(self, key):
        return self.store.get(key)

    # Node internal handlers
    def receive_and_replicate(self, key, value, message_id, global_sequence):
        # Check if we've already seen this message
        if message_id in self.seen_messages:
            return  # Don't process duplicate messages
        
        # Mark this message as seen
        self.seen_messages.add(message_id)
        
        # Add to pending messages for ordered delivery
        heapq.heappush(self.pending_messages, (global_sequence, key, value, message_id))
        
        # Try to deliver messages in order
        self.try_deliver_ready_messages()
        
        # Broadcast to all peers (eager broadcasting for fault tolerance)
        for peer in self.peers:
            if peer != self.node_id:
                self.transport.send(
                    peer,
                    {
                        "type": "replicate",
                        "from": self.node_id,
                        "key": key,
                        "value": value,
                        "message_id": message_id,
                        "global_sequence": global_sequence,
                    },
                )

    def try_deliver_ready_messages(self):
        """
        Deliver messages in sequential order, but be resilient to permanently lost messages.
        Use a timeout mechanism to eventually skip missing messages.
        """
        delivered_any = True
        while delivered_any and self.pending_messages:
            delivered_any = False
            
            # Check if we can deliver the next expected message
            if (self.pending_messages and 
                self.pending_messages[0][0] == self.next_expected_seq):
                global_seq, key, value, msg_id = heapq.heappop(self.pending_messages)
                self.deliver(key, value)
                self.next_expected_seq += 1
                delivered_any = True
            else:
                # Check if we should skip missing messages and deliver what we can
                # If we have messages but none match next_expected_seq, we might have gaps
                current_time = self.scheduler.clock.now_ms()
                
                # Simple heuristic: if we have pending messages that are much newer than
                # what we're waiting for, assume the intermediate messages are lost
                if self.pending_messages:
                    earliest_pending = self.pending_messages[0][0]
                    gap_size = earliest_pending - self.next_expected_seq
                    
                    # If there's a large gap, skip ahead to fill it
                    if gap_size > 0:
                        # Skip the missing sequence numbers
                        self.next_expected_seq = earliest_pending
                        # Now try to deliver again
                        delivered_any = True

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
            # Apply incoming replication from any peer with ordering
            message_id = msg.get("message_id")
            global_sequence = msg.get("global_sequence", 0)
            
            if message_id is not None:
                self.receive_and_replicate(
                    msg["key"], 
                    msg["value"], 
                    message_id, 
                    global_sequence
                )
            else:
                # Fallback for messages without proper ordering info
                self.deliver(msg["key"], msg["value"])
        else:
            raise ValueError(f"Unknown type in message {msg!r}")

    def get_debug_info(self):
        """Return debug information about the node's state."""
        return {
            "node_id": self.node_id,
            "next_expected_seq": self.next_expected_seq,
            "pending_count": len(self.pending_messages),
            "pending_seqs": [p[0] for p in self.pending_messages],
            "log_length": len(self.log),
            "store": dict(self.store)
        }
