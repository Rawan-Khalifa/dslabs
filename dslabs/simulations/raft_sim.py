"""Visual simulation of Raft consensus algorithm.

This simulation provides detailed logging and visualization of how Raft nodes
interact during leader election and log replication. It helps understand:

- Election timeouts and candidate transitions
- Vote requests and responses
- Leader election and heartbeats
- Log replication and commit propagation
- State transitions (FOLLOWER → CANDIDATE → LEADER)

The simulation uses fake transport and scheduler (like the unit tests) to keep
everything deterministic and easy to follow step-by-step.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time

from dslabs.algorithms.raft.raft import Raft, RaftState, LogEntry


@dataclass
class FakeTransport:
    """
    In-memory transport that logs all messages sent between nodes.
    
    This allows us to see exactly what messages are exchanged and when.
    """
    
    # Map of node_id -> handler function
    handlers: Dict[str, Callable[[Dict[str, Any]], None]] = field(default_factory=dict)
    
    # Log of all messages sent (for debugging)
    message_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def register(self, node_id: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a message handler for a node."""
        self.handlers[node_id] = handler
        print(f"[TRANSPORT] Registered handler for node {node_id}")
    
    def send(self, to: str, msg: Dict[str, Any]) -> None:
        """Send a message to a node (delivers immediately)."""
        # Log the message
        msg_type = msg.get("type", "unknown")
        from_node = msg.get("from", msg.get("leader_id", msg.get("candidate_id", "?")))
        
        self.message_log.append({
            "from": from_node,
            "to": to,
            "type": msg_type,
            "term": msg.get("term", "?"),
            "timestamp": time.time()
        })
        
        # Print the message
        if msg_type == "request_vote":
            print(f"  📨 {from_node} → {to}: RequestVote (term={msg['term']})")
        elif msg_type == "request_vote_response":
            vote = "✅ YES" if msg.get("vote_granted") else "❌ NO"
            print(f"  📨 {from_node} → {to}: VoteResponse {vote} (term={msg['term']})")
        elif msg_type == "append_entries":
            num_entries = len(msg.get("entries", []))
            if num_entries == 0:
                print(f"  💓 {from_node} → {to}: Heartbeat (term={msg['term']})")
            else:
                print(f"  📦 {from_node} → {to}: AppendEntries [{num_entries} entries] (term={msg['term']})")
        elif msg_type == "append_entries_response":
            status = "✅ SUCCESS" if msg.get("success") else "❌ FAILED"
            print(f"  📨 {from_node} → {to}: AppendEntriesResponse {status}")
        
        # Deliver the message
        if to in self.handlers:
            self.handlers[to](msg)


@dataclass
class FakeScheduler:
    """
    Fake scheduler that allows manual control of time.
    
    Instead of real timers, we collect callbacks and can trigger them manually.
    """
    
    # List of pending callbacks with their delay
    pending: List[tuple[int, Callable[[], None]]] = field(default_factory=list)
    
    # Current simulated time
    current_time: int = 0
    
    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> Callable[[], None]:
        """Schedule a callback to run after delay_ms milliseconds."""
        trigger_time = self.current_time + delay_ms
        self.pending.append((trigger_time, callback))
        
        # Return a cancel function
        def cancel():
            if (trigger_time, callback) in self.pending:
                self.pending.remove((trigger_time, callback))
        
        return cancel
    
    def advance(self, ms: int) -> None:
        """Advance simulated time and trigger any callbacks that are due."""
        print(f"\n⏰ [TIME] Advancing {ms}ms (current: {self.current_time}ms → {self.current_time + ms}ms)")
        self.current_time += ms
        
        # Find and trigger all callbacks that are now due
        due_callbacks = [(t, cb) for t, cb in self.pending if t <= self.current_time]
        
        for trigger_time, callback in due_callbacks:
            self.pending.remove((trigger_time, callback))
            print(f"  ⏰ Triggering callback (was scheduled for {trigger_time}ms)")
            callback()
    
    def get_next_timeout(self) -> Optional[int]:
        """Get the time until the next scheduled callback."""
        if not self.pending:
            return None
        next_time = min(t for t, _ in self.pending)
        return max(0, next_time - self.current_time)


def print_cluster_state(nodes: Dict[str, Raft]) -> None:
    """Print the current state of all nodes in the cluster."""
    print("\n" + "=" * 80)
    print("CLUSTER STATE")
    print("=" * 80)
    
    for node_id, node in sorted(nodes.items()):
        # State and role
        state_emoji = {
            RaftState.FOLLOWER: "👥",
            RaftState.CANDIDATE: "🗳️",
            RaftState.LEADER: "👑"
        }
        emoji = state_emoji.get(node.state, "❓")
        
        # Term and vote info
        vote_info = f"voted_for={node.voted_for}" if node.voted_for else "no vote"
        
        # Leader info
        leader_info = f"leader={node.leader_id}" if node.leader_id else "no leader"
        
        # Log info
        log_info = f"log=[{len(node.log)} entries]"
        if node.log:
            log_summary = ", ".join(f"T{e.term}:{e.command}" for e in node.log[:3])
            if len(node.log) > 3:
                log_summary += f", ... (+{len(node.log) - 3} more)"
            log_info = f"log=[{log_summary}]"
        
        # Commit info
        commit_info = f"commit_idx={node.commit_index}, last_applied={node.last_applied}"
        
        # Print node state
        print(f"\n{emoji} Node {node_id} ({node.state.value.upper()})")
        print(f"   Term: {node.current_term}, {vote_info}, {leader_info}")
        print(f"   {log_info}")
        print(f"   {commit_info}")
        
        # Leader-specific info
        if node.state == RaftState.LEADER:
            print(f"   next_index: {dict(node.next_index)}")
            print(f"   match_index: {dict(node.match_index)}")
    
    print("=" * 80 + "\n")


def scenario_leader_election():
    """
    Simulate a leader election scenario.
    
    Steps:
    1. Start 3 nodes as followers
    2. Trigger election timeout on node n1
    3. n1 becomes candidate and requests votes
    4. n2 and n3 grant votes
    5. n1 becomes leader
    6. n1 sends heartbeats
    """
    
    print("\n" + "🎬" * 40)
    print("SCENARIO: LEADER ELECTION")
    print("🎬" * 40 + "\n")
    
    # Setup
    transport = FakeTransport()
    scheduler = FakeScheduler()
    applied_commands = []
    
    def make_apply_callback(node_id: str):
        """Create an apply callback for a specific node."""
        def apply(command: Any, index: int):
            applied_commands.append((node_id, command, index))
            print(f"  ✅ [{node_id}] Applied command '{command}' at index {index}")
        return apply
    
    # Create 3 nodes
    nodes = {}
    peer_ids = ["n1", "n2", "n3"]
    
    for node_id in peer_ids:
        nodes[node_id] = Raft(
            node_id=node_id,
            peers=peer_ids,
            transport=transport,
            scheduler=scheduler,
            apply=make_apply_callback(node_id)
        )
    
    # Step 1: Start all nodes
    print("\n📍 STEP 1: Starting all nodes as followers")
    print("-" * 80)
    for node_id, node in nodes.items():
        node.start()
    
    print_cluster_state(nodes)
    
    # Step 2: Trigger election timeout on n1
    print("\n📍 STEP 2: Trigger election timeout on n1")
    print("-" * 80)
    print("⏰ n1's election timer fires (simulating timeout)")
    
    # Manually trigger n1's election
    nodes["n1"]._start_election()
    
    print_cluster_state(nodes)
    
    # Step 3: n2 grants vote to n1
    print("\n📍 STEP 3: n2 grants vote to n1")
    print("-" * 80)
    # Already happened via transport during _start_election
    
    print_cluster_state(nodes)
    
    # Step 4: n3 grants vote to n1
    print("\n📍 STEP 4: n3 grants vote to n1")
    print("-" * 80)
    # Already happened via transport during _start_election
    
    print_cluster_state(nodes)
    
    # Step 5: Advance time to trigger heartbeats
    print("\n📍 STEP 5: Leader sends heartbeats")
    print("-" * 80)
    
    next_timeout = scheduler.get_next_timeout()
    if next_timeout:
        scheduler.advance(next_timeout)
    
    print_cluster_state(nodes)
    
    # Summary
    print("\n" + "📊" * 40)
    print("SUMMARY")
    print("📊" * 40)
    print(f"Total messages sent: {len(transport.message_log)}")
    print(f"Leader elected: {nodes['n1'].leader_id}")
    print(f"Leader term: {nodes['n1'].current_term}")


def scenario_log_replication():
    """
    Simulate log replication scenario.
    
    Steps:
    1. Start 3 nodes and elect a leader
    2. Leader receives client command
    3. Leader replicates to followers
    4. Leader commits when majority acknowledges
    5. All nodes apply committed entries
    """
    
    print("\n" + "🎬" * 40)
    print("SCENARIO: LOG REPLICATION")
    print("🎬" * 40 + "\n")
    
    # Setup
    transport = FakeTransport()
    scheduler = FakeScheduler()
    applied_commands = []
    
    def make_apply_callback(node_id: str):
        def apply(command: Any, index: int):
            applied_commands.append((node_id, command, index))
            print(f"  ✅ [{node_id}] Applied command '{command}' at index {index}")
        return apply
    
    # Create 3 nodes
    nodes = {}
    peer_ids = ["n1", "n2", "n3"]
    
    for node_id in peer_ids:
        nodes[node_id] = Raft(
            node_id=node_id,
            peers=peer_ids,
            transport=transport,
            scheduler=scheduler,
            apply=make_apply_callback(node_id)
        )
    
    # Step 1: Start nodes and elect leader
    print("\n📍 STEP 1: Start nodes and elect n1 as leader")
    print("-" * 80)
    
    for node in nodes.values():
        node.start()
    
    # Manually make n1 the leader
    nodes["n1"]._start_election()
    
    # Wait for votes
    print_cluster_state(nodes)
    
    # Step 2: Client sends command to leader
    print("\n📍 STEP 2: Client sends command 'SET x=10' to leader")
    print("-" * 80)
    
    try:
        nodes["n1"].client_append("SET x=10")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    
    print_cluster_state(nodes)
    
    # Step 3: Check replication
    print("\n📍 STEP 3: Check log replication across nodes")
    print("-" * 80)
    
    print_cluster_state(nodes)
    
    # Step 4: Send another command
    print("\n📍 STEP 4: Client sends command 'SET y=20' to leader")
    print("-" * 80)
    
    try:
        nodes["n1"].client_append("SET y=20")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    
    print_cluster_state(nodes)
    
    # Summary
    print("\n" + "📊" * 40)
    print("SUMMARY")
    print("📊" * 40)
    print(f"Total messages sent: {len(transport.message_log)}")
    print(f"Commands applied: {len(applied_commands)}")
    print(f"Applied on nodes: {applied_commands}")


def main():
    """Run all simulation scenarios."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "RAFT CONSENSUS ALGORITHM SIMULATION")
    print("=" * 80)
    
    # Run scenario 1: Leader election
    scenario_leader_election()
    
    input("\n\nPress Enter to continue to log replication scenario...")
    
    # Run scenario 2: Log replication
    scenario_log_replication()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "SIMULATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()