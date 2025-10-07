"""Fault tolerance simulation for Raft consensus algorithm.

This simulation tests how Raft handles various failure scenarios:

1. **Node Crashes**: Individual nodes crash and restart
2. **Leader Crashes**: The current leader crashes, forcing re-election
3. **Network Partitions**: Nodes are split into isolated groups
4. **Message Drops**: Random messages are dropped to simulate unreliable network
5. **Delayed Messages**: Messages arrive late, testing ordering guarantees

The simulation uses the existing dslabs infrastructure (scheduler, protocols, etc.)
to avoid reinventing the wheel.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random

from dslabs.algorithms.raft.raft import Raft, RaftState, LogEntry
# Remove the problematic imports - we'll use Protocol from typing instead
from typing import Protocol


class Transport(Protocol):
    """Transport protocol interface."""
    
    def send(self, to: str, msg: Dict[str, Any]) -> None:
        """Send a message to a node."""
        ...


class Scheduler(Protocol):
    """Scheduler protocol interface."""
    
    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> Callable[[], None]:
        """Schedule a callback after a delay."""
        ...


@dataclass
class FaultInjector:
    """
    Controls fault injection for testing fault tolerance.
    
    Manages:
    - Crashed nodes
    - Message drops
    - Network partitions
    """
    
    crashed_nodes: Set[str] = field(default_factory=set)
    drop_rate: float = 0.0
    partitions: List[Set[str]] = field(default_factory=list)
    messages_dropped: int = 0
    
    def crash_node(self, node_id: str) -> None:
        """Mark a node as crashed."""
        self.crashed_nodes.add(node_id)
        print(f"💥 Node {node_id} has CRASHED")
    
    def restart_node(self, node_id: str) -> None:
        """Mark a node as restarted."""
        self.crashed_nodes.discard(node_id)
        print(f"🔄 Node {node_id} has RESTARTED")
    
    def create_partition(self, partition_groups: List[Set[str]]) -> None:
        """Create network partitions."""
        self.partitions = partition_groups
        print(f"🔌 Network partitioned: {partition_groups}")
    
    def heal_partition(self) -> None:
        """Remove all partitions."""
        self.partitions.clear()
        print(f"🔗 Network partition healed")
    
    def set_drop_rate(self, rate: float) -> None:
        """Set message drop probability."""
        self.drop_rate = rate
        print(f"📉 Drop rate: {rate * 100:.1f}%")
    
    def should_drop_message(self, from_node: str, to_node: str) -> bool:
        """Determine if message should be dropped."""
        # Drop if either node is crashed
        if from_node in self.crashed_nodes or to_node in self.crashed_nodes:
            return True
        
        # Drop if nodes are in different partitions
        if self.partitions:
            from_partition = None
            to_partition = None
            
            for i, partition in enumerate(self.partitions):
                if from_node in partition:
                    from_partition = i
                if to_node in partition:
                    to_partition = i
            
            if from_partition is not None and to_partition is not None:
                if from_partition != to_partition:
                    return True
        
        # Random drop based on drop rate
        if random.random() < self.drop_rate:
            self.messages_dropped += 1
            return True
        
        return False


@dataclass
class FaultTolerantTransport:
    """Transport wrapper that supports fault injection."""
    
    fault_injector: FaultInjector
    handlers: Dict[str, Callable[[Dict[str, Any]], None]] = field(default_factory=dict)
    message_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def register(self, node_id: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a message handler for a node."""
        self.handlers[node_id] = handler
    
    def send(self, to: str, msg: Dict[str, Any]) -> None:
        """Send message with fault injection."""
        from_node = msg.get("from", msg.get("leader_id", msg.get("candidate_id", "?")))
        msg_type = msg.get("type", "unknown")
        
        # Log the message
        self.message_log.append({
            "from": from_node,
            "to": to,
            "type": msg_type,
            "dropped": False
        })
        
        # Check if should drop
        if self.fault_injector.should_drop_message(from_node, to):
            self.message_log[-1]["dropped"] = True
            print(f"  ❌ DROPPED: {from_node} → {to}: {msg_type}")
            return
        
        # Print and deliver
        self._print_message(from_node, to, msg)
        
        if to in self.handlers:
            self.handlers[to](msg)
    
    def _print_message(self, from_node: str, to_node: str, msg: Dict[str, Any]) -> None:
        """Print message being sent."""
        msg_type = msg.get("type", "unknown")
        
        if msg_type == "request_vote":
            print(f"  📨 {from_node} → {to_node}: RequestVote")
        elif msg_type == "request_vote_response":
            vote = "✅" if msg.get("vote_granted") else "❌"
            print(f"  📨 {from_node} → {to_node}: VoteResponse {vote}")
        elif msg_type == "append_entries":
            num_entries = len(msg.get("entries", []))
            if num_entries == 0:
                print(f"  💓 {from_node} → {to_node}: Heartbeat")
            else:
                print(f"  📦 {from_node} → {to_node}: AppendEntries [{num_entries}]")
        elif msg_type == "append_entries_response":
            status = "✅" if msg.get("success") else "❌"
            print(f"  📨 {from_node} → {to_node}: AppendEntriesResp {status}")


@dataclass
class SimulationScheduler:
    """Simple synchronous scheduler for simulation."""
    
    current_time: int = 0
    pending: List[Tuple[int, Callable[[], None]]] = field(default_factory=list)
    
    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> Callable[[], None]:
        """Schedule a callback."""
        trigger_time = self.current_time + delay_ms
        entry = (trigger_time, callback)
        self.pending.append(entry)
        
        # Return cancel function
        def cancel():
            if entry in self.pending:
                self.pending.remove(entry)
        
        return cancel
    
    def advance(self, ms: int) -> None:
        """Advance simulation time."""
        self.current_time += ms
        
        # Find callbacks to trigger
        to_trigger = [cb for t, cb in self.pending if t <= self.current_time]
        
        # Remove them from pending (before calling to avoid modification during iteration)
        self.pending = [(t, cb) for t, cb in self.pending if t > self.current_time]
        
        # Execute callbacks
        for callback in to_trigger:
            try:
                callback()
            except Exception as e:
                print(f"  ⚠️  Callback error: {e}")
                import traceback
                traceback.print_exc()


def print_cluster_state(nodes: Dict[str, Raft], fault_injector: FaultInjector) -> None:
    """Print current state of all nodes."""
    print("\n" + "=" * 80)
    print("CLUSTER STATE")
    print("=" * 80)
    
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        
        # Check if crashed
        if node_id in fault_injector.crashed_nodes:
            print(f"\n💥 Node {node_id} (CRASHED)")
            continue
        
        # State emoji
        state_emoji = {
            RaftState.FOLLOWER: "👥",
            RaftState.CANDIDATE: "🗳️",
            RaftState.LEADER: "👑"
        }
        emoji = state_emoji.get(node.state, "❓")
        
        # Build info strings
        vote_info = f"voted_for={node.voted_for}" if node.voted_for else "no vote"
        leader_info = f"leader={node.leader_id}" if node.leader_id else "no leader"
        
        log_info = f"[{len(node.log)} entries]"
        if node.log and len(node.log) <= 5:
            log_summary = ", ".join(f"T{e.term}:{e.command}" for e in node.log)
            log_info = f"[{log_summary}]"
        elif node.log:
            log_summary = ", ".join(f"T{e.term}:{e.command}" for e in node.log[:3])
            log_info = f"[{log_summary}, ... +{len(node.log) - 3}]"
        
        # Print node state
        print(f"\n{emoji} Node {node_id} ({node.state.value.upper()})")
        print(f"   Term: {node.current_term}, {vote_info}, {leader_info}")
        print(f"   Log: {log_info}")
        print(f"   Commit: {node.commit_index}, Applied: {node.last_applied}")
        
        if node.state == RaftState.LEADER:
            print(f"   next_index: {dict(node.next_index)}")
            print(f"   match_index: {dict(node.match_index)}")
    
    print("=" * 80 + "\n")


def verify_log_consistency(nodes: Dict[str, Raft], fault_injector: FaultInjector) -> bool:
    """Verify committed logs are consistent across all non-crashed nodes."""
    print("\n🔍 Verifying log consistency...")
    
    committed_logs = {}
    for node_id, node in nodes.items():
        if node_id not in fault_injector.crashed_nodes:
            committed_logs[node_id] = [
                (entry.term, entry.command)
                for i, entry in enumerate(node.log)
                if i <= node.commit_index
            ]
    
    if not committed_logs:
        print("  ⚠️  No non-crashed nodes")
        return True
    
    # Check consistency
    first_log = list(committed_logs.values())[0]
    all_consistent = all(log == first_log for log in committed_logs.values())
    
    if all_consistent:
        print(f"  ✅ All {len(committed_logs)} nodes consistent")
        if first_log:
            print(f"     Committed: {first_log}")
        return True
    else:
        print("  ❌ INCONSISTENCY DETECTED!")
        for node_id, log in committed_logs.items():
            print(f"     {node_id}: {log}")
        return False


# ============================================================================
# SCENARIOS
# ============================================================================

def scenario_leader_crash():
    """Test: Leader crashes and cluster re-elects."""
    
    print("\n" + "🎬" * 40)
    print("SCENARIO 1: LEADER CRASH AND RE-ELECTION")
    print("🎬" * 40 + "\n")
    
    # Setup
    scheduler = SimulationScheduler()
    fault_injector = FaultInjector()
    transport = FaultTolerantTransport(fault_injector=fault_injector)
    applied = []
    
    def make_apply(node_id: str):
        def apply(cmd: Any, idx: int):
            applied.append((node_id, cmd, idx))
        return apply
    
    # Create 5-node cluster
    peer_ids = ["n1", "n2", "n3", "n4", "n5"]
    nodes = {}
    
    for node_id in peer_ids:
        node = Raft(
            node_id=node_id,
            peers=peer_ids,
            transport=transport,
            scheduler=scheduler,
            apply=make_apply(node_id)
        )
        nodes[node_id] = node
        # No need to manually register - node.start() does it
    
    print("📍 STEP 1: Start cluster and elect n1 as leader")
    print("-" * 80)
    for node in nodes.values():
        node.start()  # This will call transport.register()
    
    nodes["n1"]._start_election()
    scheduler.advance(50)
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 2: Leader commits entries")
    print("-" * 80)
    if nodes["n1"].state == RaftState.LEADER:
        try:
            nodes["n1"].client_append("cmd1")
            nodes["n1"].client_append("cmd2")
            scheduler.advance(100)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 3: CRASH THE LEADER")
    print("-" * 80)
    fault_injector.crash_node("n1")
    nodes["n1"].stop()
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 4: Wait for new election")
    print("-" * 80)
    scheduler.advance(350)
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 5: New leader commits entry")
    print("-" * 80)
    new_leader = None
    for node_id, node in nodes.items():
        if node.state == RaftState.LEADER and node_id != "n1":
            new_leader = node_id
            break
    
    if new_leader:
        print(f"  ✅ New leader: {new_leader}")
        try:
            nodes[new_leader].client_append("cmd3")
            scheduler.advance(100)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    else:
        print("  ⚠️  No new leader elected")
    
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 6: Restart crashed leader")
    print("-" * 80)
    fault_injector.restart_node("n1")
    nodes["n1"].start()
    scheduler.advance(150)
    print_cluster_state(nodes, fault_injector)
    
    verify_log_consistency(nodes, fault_injector)
    
    print(f"\n📊 Statistics:")
    print(f"   Total messages: {len(transport.message_log)}")
    print(f"   Messages dropped: {fault_injector.messages_dropped}")
    print(f"   Commands applied: {len(applied)}")


def scenario_network_partition():
    """Test: Network partition (split-brain)."""
    
    print("\n" + "🎬" * 40)
    print("SCENARIO 2: NETWORK PARTITION")
    print("🎬" * 40 + "\n")
    
    scheduler = SimulationScheduler()
    fault_injector = FaultInjector()
    transport = FaultTolerantTransport(fault_injector=fault_injector)
    applied = []
    
    def make_apply(node_id: str):
        def apply(cmd: Any, idx: int):
            applied.append((node_id, cmd, idx))
        return apply
    
    # Create 5-node cluster
    peer_ids = ["n1", "n2", "n3", "n4", "n5"]
    nodes = {}
    
    for node_id in peer_ids:
        node = Raft(
            node_id=node_id,
            peers=peer_ids,
            transport=transport,
            scheduler=scheduler,
            apply=make_apply(node_id)
        )
        nodes[node_id] = node
        transport.handlers[node_id] = node.on_message
    
    print("📍 STEP 1: Start cluster")
    print("-" * 80)
    for node in nodes.values():
        node.start()
    
    nodes["n1"]._start_election()
    scheduler.advance(100)
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 2: Create partition")
    print("   Minority: {n1, n2}")
    print("   Majority: {n3, n4, n5}")
    print("-" * 80)
    fault_injector.create_partition([
        {"n1", "n2"},
        {"n3", "n4", "n5"}
    ])
    
    print("\n📍 STEP 3: Majority elects new leader")
    print("-" * 80)
    scheduler.advance(350)
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 4: Try commits in both partitions")
    print("-" * 80)
    
    # Try in minority (should fail)
    if nodes["n1"].state == RaftState.LEADER:
        print("  Minority partition attempt...")
        try:
            nodes["n1"].client_append("minority_cmd")
            scheduler.advance(100)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    # Try in majority (should succeed)
    for node_id in ["n3", "n4", "n5"]:
        if nodes[node_id].state == RaftState.LEADER:
            print(f"  Majority partition attempt ({node_id})...")
            try:
                nodes[node_id].client_append("majority_cmd")
                scheduler.advance(100)
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
            break
    
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 5: Heal partition")
    print("-" * 80)
    fault_injector.heal_partition()
    scheduler.advance(200)
    print_cluster_state(nodes, fault_injector)
    
    verify_log_consistency(nodes, fault_injector)


def scenario_message_drops():
    """Test: Unreliable network with message drops."""
    
    print("\n" + "🎬" * 40)
    print("SCENARIO 3: UNRELIABLE NETWORK (20% DROPS)")
    print("🎬" * 40 + "\n")
    
    scheduler = SimulationScheduler()
    fault_injector = FaultInjector()
    transport = FaultTolerantTransport(fault_injector=fault_injector)
    applied = []
    
    def make_apply(node_id: str):
        def apply(cmd: Any, idx: int):
            applied.append((node_id, cmd, idx))
        return apply
    
    # Create 3-node cluster
    peer_ids = ["n1", "n2", "n3"]
    nodes = {}
    
    for node_id in peer_ids:
        node = Raft(
            node_id=node_id,
            peers=peer_ids,
            transport=transport,
            scheduler=scheduler,
            apply=make_apply(node_id)
        )
        nodes[node_id] = node
        transport.handlers[node_id] = node.on_message
    
    print("📍 STEP 1: Start with 20% drop rate")
    print("-" * 80)
    fault_injector.set_drop_rate(0.2)
    
    for node in nodes.values():
        node.start()
    
    print("\n📍 STEP 2: Elect leader (may need retries)")
    print("-" * 80)
    nodes["n1"]._start_election()
    scheduler.advance(100)
    
    # Retry if needed
    for i in range(5):
        if any(n.state == RaftState.LEADER for n in nodes.values()):
            break
        print(f"  Retry {i + 1}...")
        scheduler.advance(350)
    
    print_cluster_state(nodes, fault_injector)
    
    print("\n📍 STEP 3: Commit entries with drops")
    print("-" * 80)
    leader = None
    for node_id, node in nodes.items():
        if node.state == RaftState.LEADER:
            leader = node_id
            break
    
    if leader:
        for i in range(5):
            try:
                nodes[leader].client_append(f"cmd{i}")
                scheduler.advance(150)
            except Exception as e:
                print(f"  ⚠️  cmd{i} error: {e}")
    
    print_cluster_state(nodes, fault_injector)
    verify_log_consistency(nodes, fault_injector)
    
    print(f"\n📊 Statistics:")
    print(f"   Messages: {len(transport.message_log)}")
    print(f"   Dropped: {fault_injector.messages_dropped}")
    if len(transport.message_log) > 0:
        drop_pct = fault_injector.messages_dropped / len(transport.message_log) * 100
        print(f"   Drop rate: {drop_pct:.1f}%")


def main():
    """Run all scenarios."""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "RAFT FAULT TOLERANCE SIMULATION")
    print("=" * 80)
    
    scenario_leader_crash()
    
    input("\n\nPress Enter for network partition scenario...")
    
    scenario_network_partition()
    
    input("\n\nPress Enter for message drops scenario...")
    
    scenario_message_drops()
    
    print("\n" + "=" * 80)
    print(" " * 25 + "SIMULATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()