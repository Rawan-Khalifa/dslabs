"""Guidance and scaffolding for implementing the Raft consensus algorithm.

This module mirrors the structure described in *Raft: In Search of an
Understandable Consensus Algorithm* (Ongaro & Ousterhout, 2014). The goal is to
provide you with explicit hooks, rich documentation, and light-weight
scaffolding so that the implementation can focus on the algorithmic ideas:
leader election, log replication, safety, and the interaction with application
state machines.

You should:

* Read the Raft paper and map each major concept to the methods declared here.
* Rely on `dslabs.protocols.Transport` for network I/O and `dslabs.protocols.Scheduler`
  for timers; the unit tests in tests/test_raft_algorithm.py inject fake
  implementations of these protocols to keep the logic deterministic.
* Follow the provided docstrings and inline comments as a step-by-step outline
  when filling in each method. The comments are not exhaustive, but they call
  out important conditions, state transitions, and message flows that must be handled.

Until the algorithm is implemented, the stubs intentionally raise `NotImplementedError`
so that the unit tests fail, reminding you to finish the implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set  # ✅ Added Set here

from dslabs.protocols import Scheduler, SchedulerCancel, Transport
import random

class RaftState(str, Enum):
    """
    High-level role assumed by a Raft node.

    Raft rotates between three roles:

    ``FOLLOWER``
        Passive role, responds to requests from leaders or candidates and resets
        its election timeout when heartbeats arrive.

    ``CANDIDATE``
        Initiated after an election timeout; the node increments its term,
        votes for itself, and requests votes from peers in pursuit of
        leadership.

    ``LEADER``
        The node responsible for log replication and serving client requests.
        Leaders send periodic AppendEntries heartbeats to maintain authority.
    """

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """
    Single entry stored within the replicated log.

    Parameters
    ----------
    term:
        Election term under which the entry was created by the leader. Raft
        relies on term comparisons to uphold the *log matching* property.
    command:
        Opaque client command that should be applied to the state machine once
        committed. The implementation decides the structure (dict, tuple, etc.).
    """

    term: int
    command: Any


@dataclass
class Raft:
    """
    Skeleton of the Raft state machine.

    Parameters:
    ----------
    node_id:
        Identifier for this node. Raft messages carry string identifiers and
        the tests use human-readable values ("n1", "n2", etc.).
    peers:
        List of node identifiers participating in the cluster. The local node
        may or may not appear in this list, depending on how the runner builds
        membership; the implementation should be robust to either choice.
    transport:
        Implementation of `dslabs.protocols.Transport` used to send Raft
        protocol messages. You must call `transport.send` with JSON-like
        dictionaries describing RequestVote and AppendEntries interactions.
    scheduler:
        Implementation of `dslabs.protocols.Scheduler` that provides election
        timeouts and heartbeat intervals. Timers are critical to Raft’s
        liveness guarantees.
    apply:
        Callback invoked with ``(command, index)`` whenever an entry becomes
        committed and should be applied to the replicated state machine. The
        tests assert on this hook to check that commits are signalled correctly.

    Attributes
    ----------
    state:
        Current `RaftState`. Start as follower, transition per the paper.
    current_term / voted_for:
        Persistent election metadata. `current_term` increments on new elections;
        `voted_for` tracks which candidate received our vote in the current term
        (or `None` if no vote was cast yet).
    log:
        In-memory log containing `LogEntry` entries. Index 0 corresponds to the
        first command appended by any leader.
    commit_index / last_applied:
        Match the definitions from the paper. `commit_index` tracks the highest
        log index known to be committed; `last_applied` is the highest index
        already delivered to `apply`.
    next_index / match_index:
        Leader-only replication metadata. `next_index` is the next log index
        that should be sent to each follower; `match_index` stores the highest
        index known to be replicated on each follower.
    leader_id:
        Convenience field to remember the current leader (useful for followers
        redirecting client requests).
    _election_timer / _heartbeat_timer:
        Handles returned by `scheduler.call_later` so timers can be cancelled
        or reset. Private because the tests do not rely on their exact type.
    """

    node_id: str
    peers: List[str]
    transport: Transport
    scheduler: Scheduler
    apply: Callable[[Any, int], None]
    state: RaftState = field(default=RaftState.FOLLOWER, init=False)
    current_term: int = field(default=0, init=False)
    voted_for: Optional[str] = field(default=None, init=False)
    log: List[LogEntry] = field(default_factory=list, init=False)
    commit_index: int = field(default=-1, init=False)
    last_applied: int = field(default=-1, init=False)
    next_index: Dict[str, int] = field(default_factory=dict, init=False)
    match_index: Dict[str, int] = field(default_factory=dict, init=False)
    leader_id: Optional[str] = field(default=None, init=False)
    _election_timer: Optional[SchedulerCancel] = field(default=None, init=False)
    _heartbeat_timer: Optional[SchedulerCancel] = field(default=None, init=False)
    _votes_received: Set[str] = field(default_factory=set, init=False)  # ✅ Correct!

    def start(self) -> None:
        """
        Prepare the node for participation in Raft.

        Responsibilities (see Section 5.2 of the paper):

        #. Register the `on_message` handler with the transport so that
           inbound RPCs are delivered to this instance.
        #. Reset state as necessary (e.g., ensure leader-specific maps are
           cleared when starting as a follower).
        #. Schedule a randomized election timeout via `_reset_election_timer`
           so the node eventually transitions to a candidate if no leader is
           heard from.

        Suggested implementation sketch:

           self.transport.register(self.node_id, self.on_message)
           self.state = RaftState.FOLLOWER
           self.leader_id = None
           self._reset_election_timer()

        The concrete steps may differ, but capturing these responsibilities is
        essential to bootstrapping the node. The tests will fail until the logic
        meets the documented expectations.
        """

        #TODO:
        # 1) Register the on_message handler with the transport
        # This ensures inbound RPCs are delivered to this Raft instance
        self.transport.register(self.node_id, self.on_message)
        
        # 2) Start as FOLLOWER
        self.state = RaftState.FOLLOWER #used the class RaftState 
        
        # When starting as a follower, we don't know who the leader is yet
        self.leader_id = None
        
        # Also clear leader-specific replication metadata (next_index, match_index)
        # These are only used when we're the leader
        self.next_index.clear()
        self.match_index.clear()
        
        # 3) Schedule a randomized election timeout
        # This will eventually cause us to become a candidate if no leader is heard from
        self._reset_election_timer()


    def client_append(self, command: Any) -> None:
        """
        Append a client command to the replicated log.

        Only the leader should accept client writes. Followers should direct
        clients to the known leader by returning or forwarding the request.

        Expected workflow when this node is the leader:

        #. Append a `LogEntry` containing `(current_term, command)` to the local
           log.
        #. Update `next_index` / `match_index` bookkeeping if this is the first
           entry or if peers lag behind.
        #. Immediately send AppendEntries RPCs (heartbeats with payloads) to all
           followers so replication proceeds without waiting for the next
           periodic heartbeat. The payload should carry `prev_log_index`,
           `prev_log_term`, `entries`, and `leader_commit` as described in the
           paper.

        The helper method should raise an error or ignore commands when the node
        is not the leader; the exact behaviour can be tailored to the runtime
        but should be consistent.
        """

        # TODO:
        #  * check whether or not in the leader role
        #  * append entry
        #  * send AppendEntries to followers
            # Only leaders can accept client commands
        if self.state != RaftState.LEADER:
            # Could raise an error or redirect to leader
            # For now, we'll just ignore (tests might expect an error)
            raise RuntimeError(f"Not the leader. Leader is {self.leader_id}")
        
        # Step 1: Append entry to local log
        new_entry = LogEntry(term=self.current_term, command=command)
        self.log.append(new_entry)
        
        # Step 2: Update our own match_index (we have the entry locally)
        self.match_index[self.node_id] = len(self.log) - 1
        
        # Step 3: Immediately replicate to followers (don't wait for heartbeat)
        self._send_heartbeats()

    def _handle_request_vote(self, msg: Dict[str, Any]) -> None:
        """
        Process a RequestVote RPC from a candidate.
        
        Per Section 5.2 and 5.4 of the paper:
        - Grant vote if:
        1. Haven't voted in this term (or already voted for this candidate)
        2. Candidate's log is at least as up-to-date as ours
        - Deny vote otherwise
        
        Reply with vote_granted=True/False and current term.
        """
        
        candidate_id = msg.get("candidate_id")
        candidate_term = msg.get("term")
        candidate_last_log_index = msg.get("last_log_index", -1)
        candidate_last_log_term = msg.get("last_log_term", 0)
        
        # Determine our last log info
        our_last_log_index = len(self.log) - 1
        our_last_log_term = self.log[our_last_log_index].term if self.log else 0
        
        # Check if candidate's log is at least as up-to-date as ours (Section 5.4.1)
        # Log is more up-to-date if:
        # - Last entry has higher term, OR
        # - Same term but longer log
        log_is_up_to_date = (
            candidate_last_log_term > our_last_log_term or
            (candidate_last_log_term == our_last_log_term and 
            candidate_last_log_index >= our_last_log_index)
        )
        
        # Grant vote if:
        # 1. Haven't voted yet OR already voted for this candidate
        # 2. Candidate's log is up-to-date
        vote_granted = False
        if (self.voted_for is None or self.voted_for == candidate_id) and log_is_up_to_date:
            vote_granted = True
            self.voted_for = candidate_id
            # Reset election timer since we granted a vote
            self._reset_election_timer()
        
        # Send response
        self.transport.send(candidate_id, {
            "type": "request_vote_response",
            "term": self.current_term,
            "vote_granted": vote_granted,
            "from": self.node_id
        })


    def _handle_request_vote_response(self, msg: Dict[str, Any]) -> None:
        """
        Process vote responses when we're a candidate.
        
        Per Section 5.2:
        - Count votes received
        - If majority grants votes → become leader
        - If term in response > current_term → step down to follower
        """
        
        # Only candidates care about vote responses
        if self.state != RaftState.CANDIDATE:
            return
        
        # If response is from an old term, ignore it
        response_term = msg.get("term", 0)
        if response_term < self.current_term:
            return
        
        vote_granted = msg.get("vote_granted", False)
        from_node = msg.get("from")
    
        if vote_granted and from_node:
            # Track this vote
            self._votes_received.add(from_node)
            
            # Count total votes (including our own vote for ourselves)
            total_votes = len(self._votes_received) + 1  # +1 for voting for ourselves
            
            # Calculate majority
            majority = (len(self.peers) + 1) // 2 + 1
            
            # If we have majority, become leader
            if total_votes >= majority:
                self._become_leader()

    def _handle_append_entries(self, msg: Dict[str, Any]) -> None:
        """
        Process an AppendEntries RPC from the leader.
        
        Per Section 5.2 and 5.3 of the paper:
        - Reset election timer (valid leader contact)
        - Check if our log matches at prev_log_index
        - Append new entries if log is consistent
        - Update commit_index if leader's commit is higher
        - Reply with success/failure
        """
        
        leader_id = msg.get("leader_id")
        leader_term = msg.get("term")
        prev_log_index = msg.get("prev_log_index", -1)
        prev_log_term = msg.get("prev_log_term", 0)
        entries = msg.get("entries", [])
        leader_commit = msg.get("leader_commit", -1)
        
        # If we're a candidate and receive AppendEntries from valid leader, step down
        if self.state == RaftState.CANDIDATE and leader_term >= self.current_term:
            self.state = RaftState.FOLLOWER
        
        # Update our knowledge of who the leader is
        if leader_term >= self.current_term:
            self.leader_id = leader_id
            # Reset election timer - we heard from the leader
            self._reset_election_timer()
        
        # Reply false if term < currentTerm (Section 5.1)
        success = False
        if leader_term < self.current_term:
            self.transport.send(leader_id, {
                "type": "append_entries_response",
                "term": self.current_term,
                "success": False,
                "from": self.node_id
            })
            return
        
        # Check log consistency (Section 5.3)
        # Reply false if log doesn't contain an entry at prev_log_index
        # whose term matches prev_log_term
        if prev_log_index >= 0:
            if prev_log_index >= len(self.log):
                # Log is too short
                success = False
            elif self.log[prev_log_index].term != prev_log_term:
                # Term mismatch at prev_log_index
                success = False
                # Delete conflicting entry and all that follow (Section 5.3)
                self.log = self.log[:prev_log_index]
            else:
                # Log matches at prev_log_index
                success = True
        else:
            # prev_log_index == -1, so this is the first entry or empty heartbeat
            success = True
        
        # If log is consistent, append new entries
        if success and entries:
            # Find the insertion point
            insert_index = prev_log_index + 1
            
            for i, entry_dict in enumerate(entries):
                log_index = insert_index + i
                new_entry = LogEntry(term=entry_dict["term"], command=entry_dict["command"])
                
                if log_index < len(self.log):
                    # If existing entry conflicts, delete it and all that follow (Section 5.3)
                    if self.log[log_index].term != new_entry.term:
                        self.log = self.log[:log_index]
                        self.log.append(new_entry)
                else:
                    # Append new entry
                    self.log.append(new_entry)
        
        # Update commit_index if leader's commit is higher
        if leader_commit > self.commit_index:
            # Set commit_index = min(leader_commit, index of last new entry)
            self.commit_index = min(leader_commit, len(self.log) - 1)
            
            # Apply newly committed entries to state machine
            self._apply_committed_entries()
        
        # Send response
        self.transport.send(leader_id, {
            "type": "append_entries_response",
            "term": self.current_term,
            "success": success,
            "match_index": len(self.log) - 1 if success else -1,
            "from": self.node_id
        })

    def _apply_committed_entries(self) -> None:
        """
        Apply committed log entries to the state machine.
        
        Called when commit_index is updated.
        Applies all entries from last_applied + 1 to commit_index.
        """
        
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            
            # Call the apply callback with (command, index)
            self.apply(entry.command, self.last_applied)

    def _handle_append_entries_response(self, msg: Dict[str, Any]) -> None:
        """
        Process AppendEntries response from a follower.
        
        Per Section 5.3:
        - If successful, update next_index and match_index for follower
        - If failed due to log inconsistency, decrement next_index and retry
        - Check if we can advance commit_index (majority replication)
        """
        
        # Only leaders care about AppendEntries responses
        if self.state != RaftState.LEADER:
            return
        
        from_node = msg.get("from")
        response_term = msg.get("term", 0)
        success = msg.get("success", False)
        match_index = msg.get("match_index", -1)
        
        # If response is from old term, ignore it
        if response_term < self.current_term:
            return
        
        if success:
            # Update next_index and match_index for this follower
            self.match_index[from_node] = match_index
            self.next_index[from_node] = match_index + 1
            
            # Check if we can advance commit_index
            # Find highest N where majority of match_index[i] >= N
            # and log[N].term == current_term
            self._update_commit_index()
        else:
            # Log inconsistency - decrement next_index and retry
            if from_node in self.next_index:
                self.next_index[from_node] = max(0, self.next_index[from_node] - 1)
                
                # Immediately retry with updated next_index
                # (This will be sent on next heartbeat, or we can send immediately)
                self._send_append_entries_to_peer(from_node)

    def _update_commit_index(self) -> None:
        """
        Update commit_index based on match_index values from followers.
        
        Per Section 5.3 and 5.4:
        - Find highest N where majority of match_index[i] >= N
        - Only commit entries from current term
        """
        
        if self.state != RaftState.LEADER:
            return
        
        # For each possible index N from commit_index + 1 to end of log
        for n in range(self.commit_index + 1, len(self.log)):
            # Only commit entries from current term (Section 5.4.2)
            if self.log[n].term != self.current_term:
                continue
            
            # Count how many nodes have replicated this entry
            replicated_count = 1  # Count ourselves
            for peer in self.peers:
                if peer != self.node_id:
                    if self.match_index.get(peer, -1) >= n:
                        replicated_count += 1
            
            # Check if majority has replicated
            majority = (len(self.peers) + 1) // 2 + 1
            if replicated_count >= majority:
                self.commit_index = n
                # Apply newly committed entries
                self._apply_committed_entries()

    def _send_append_entries_to_peer(self, peer: str) -> None:
        """
        Send AppendEntries RPC to a specific peer.
        Helper for immediate retries when log inconsistency detected.
        """
        
        if self.state != RaftState.LEADER:
            return
        
        next_idx = self.next_index.get(peer, len(self.log))
        prev_log_index = next_idx - 1
        prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
        
        entries = []
        if next_idx < len(self.log):
            entries = [
                {"term": entry.term, "command": entry.command}
                for entry in self.log[next_idx:]
            ]
        
        self.transport.send(peer, {
            "type": "append_entries",
            "term": self.current_term,
            "leader_id": self.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": self.commit_index
        })   

    def _become_leader(self) -> None:
        """
        Transition to leader after winning election.
        
        Per Section 5.2:
        - Cancel election timer
        - Initialize next_index and match_index for all followers
        - Start sending heartbeats immediately
        """
        
        # Only candidates should become leaders
        if self.state != RaftState.CANDIDATE:
            return
        
        # Transition to leader
        self.state = RaftState.LEADER
        self.leader_id = self.node_id
        
        # Cancel election timer (leaders don't need it)
        self._cancel_election_timer()
        
        # Initialize leader state (Section 5.3)
        # next_index: for each server, index of next log entry to send
        # (initialized to leader's last log index + 1)
        last_log_index = len(self.log)
        for peer in self.peers:
            if peer != self.node_id:
                self.next_index[peer] = last_log_index
                self.match_index[peer] = -1
        
        # Start sending heartbeats immediately
        self._send_heartbeats()
        self._reset_heartbeat_timer()

    def _send_heartbeats(self) -> None:
        """
        Send AppendEntries RPCs (heartbeats) to all followers.
        
        Leaders send these periodically to:
        - Maintain authority (prevent elections)
        - Replicate log entries
        """
        
        # Only leaders send heartbeats
        if self.state != RaftState.LEADER:
            return
        
        for peer in self.peers:
            if peer != self.node_id:
                # Determine what to send based on next_index
                next_idx = self.next_index.get(peer, len(self.log))
                
                # Previous log entry info (for consistency check)
                prev_log_index = next_idx - 1
                prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
                
                # Entries to send (empty for heartbeat, or actual entries if replicating)
                entries = []
                if next_idx < len(self.log):
                    entries = [
                        {"term": entry.term, "command": entry.command}
                        for entry in self.log[next_idx:]
                    ]
                
                # Send AppendEntries RPC
                self.transport.send(peer, {
                    "type": "append_entries",
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "prev_log_index": prev_log_index,
                    "prev_log_term": prev_log_term,
                    "entries": entries,
                    "leader_commit": self.commit_index
                })


    def on_message(self, msg: Dict[str, Any]) -> None:
        """
        Dispatch inbound Raft RPCs to the appropriate handler.

        Raft exchanges two primary message types:

        `request_vote` / `request_vote_response`
            Used during elections. Followers decide whether to grant votes; the
            candidate tallies responses to determine leadership.

        `append_entries` / `append_entries_response`
            Leaders use AppendEntries for both heartbeats (empty `entries`)
            and log replication (one or more `LogEntry` records).

        Implementation outline:

        #. Inspect `msg["type"]` and branch accordingly.
        #. Handle term comparisons first: if the incoming `term` is greater
           than `current_term` the node must step down to follower and update
           `current_term` (Raft guarantees are rooted in monotonic term
           numbers).
        #. Delegate to helper methods such as `_handle_request_vote` or
           `_handle_append_entries` that you should implement.
        #. Ensure election timers are reset on valid leader activity and that
           responses get sent using `transport.send`.

        Following the structure in Figure 2 of the Raft paper makes the logic
        manageable. Thorough logging and comments often help with debugging.
        """

        # TODO:
        #  * branch on message type
        #  * apply term rules
        #  * call role-specific handlers
        
        # Step 1: Extract message type and term
        msg_type = msg.get("type")
        msg_term = msg.get("term", 0)
        
        # Step 2: Apply term rules (Section 5.1 of paper)
        # If RPC request or response contains term T > currentTerm:
        # set currentTerm = T, convert to follower
        if msg_term > self.current_term:
            self.current_term = msg_term
            self.voted_for = None
            
            # If we were leader or candidate, step down to follower
            if self.state != RaftState.FOLLOWER:
                self.state = RaftState.FOLLOWER
                self.leader_id = None
                self._cancel_heartbeat_timer()
            
            # Reset election timer since we heard from a node with higher term
            self._reset_election_timer()
        
        # Step 3: Branch on message type and delegate to handlers
        if msg_type == "request_vote":
            self._handle_request_vote(msg)
        
        elif msg_type == "request_vote_response":
            self._handle_request_vote_response(msg)
        
        elif msg_type == "append_entries":
            self._handle_append_entries(msg)
        
        elif msg_type == "append_entries_response":
            self._handle_append_entries_response(msg)
        
        else:
            # Unknown message type - ignore or log
            pass
        

    def stop(self) -> None:
        """
        Clean up timers and prepare the node to shut down or restart.

        Raft nodes may need to pause (e.g., when leaving a simulation or
        stepping down in tests). A minimal implementation should:

        #. Cancel outstanding election and heartbeat timers using the private
           helpers below.
        #. Optionally flush leader metadata so a later `start` call begins
           from the follower role with a fresh timeout.

        The function does not need to persist state; that responsibility lives
        with higher-level components if durability is desired.
        """

        # TODO:
        #  * cancel timers
        #  * reset transient leader state
            # Cancel all timers
        self._cancel_election_timer()
        self._cancel_heartbeat_timer()
        
        # Reset to follower state for clean restart
        self.state = RaftState.FOLLOWER
        self.leader_id = None
        
        # Clear leader-specific metadata
        self.next_index.clear()
        self.match_index.clear()

    # Helper hooks left for future implementation
    def _reset_election_timer(self) -> None:
        """
        Schedule the next election timeout.

        Requirements captured in Section 5.2 of the Raft paper:

        * Randomize the timeout between `T` and `2T` (or similar) to reduce the
          chance of split votes. Use the injected `scheduler` to register a
          callback that triggers the election routine.
        * Cancel any existing election timer before scheduling a new one to
          avoid duplicate callbacks firing.
        * The callback should transition the node to candidate (if still a
          follower) and initiate vote requests.

        The tests observe that a timer is scheduled, but do not mandate the
        exact randomness distribution. You can choose appropriate constants.
        """

        # TODO:
        #  * cancel old timer
        self._cancel_election_timer()

        #  * compute randomized delay
        # Using typical Raft values: 150-300ms election timeout
        min_timeout = 150  # T (minimum election timeout in ms)
        max_timeout = 300  # 2T (maximum election timeout in ms)
        timeout_ms = random.randint(min_timeout, max_timeout)

        #  * schedule election callback
        self._election_timer = self.scheduler.call_later(
            timeout_ms,
            self._start_election
        )
    
    def _start_election(self) -> None:
        """
        Transition to candidate and initiate an election.
        
        This method is called when the election timeout fires.
        
        Per Section 5.2 of the Raft paper:
        1. Increment current_term
        2. Transition to CANDIDATE state
        3. Vote for self
        4. Reset election timer
        5. Send RequestVote RPCs to all peers
        
        If we receive votes from a majority, we become the leader.
        If another candidate becomes leader, we step down.
        If election timeout elapses, start a new election.
        """
        
        # Only followers and candidates should start elections
        # (Leaders don't need to - they're already the leader)
        if self.state == RaftState.LEADER:
            return
    
        # Increment term for the new election
        self.current_term += 1
        
        # Transition to CANDIDATE
        self.state = RaftState.CANDIDATE
        
        # Vote for ourselves
        self.voted_for = self.node_id
        
        # Clear previous votes and track our own vote
        self._votes_received.clear()  # ✅ Reset vote tracking
        self._votes_received.add(self.node_id)  # ✅ Count our own vote
        
        # Reset election timer
        self._reset_election_timer()
        
        # Send RequestVote RPC to all peers
        last_log_index = len(self.log) - 1
        last_log_term = self.log[last_log_index].term if self.log else 0
        
        for peer in self.peers:
            if peer != self.node_id:
                self.transport.send(peer, {
                    "type": "request_vote",
                    "term": self.current_term,
                    "candidate_id": self.node_id,
                    "last_log_index": last_log_index,
                    "last_log_term": last_log_term
                })

    def _cancel_election_timer(self) -> None:
        """
        Stop the currently scheduled election timeout, if any.
        """

        # TODO:
        #  * call the stored cancel handle and clear it
        if self._election_timer is not None:
            self._election_timer()  # Call the cancel handle
            self._election_timer = None

    def _reset_heartbeat_timer(self) -> None:
        """
        Schedule the next heartbeat for leaders.

        Heartbeats are simply AppendEntries RPCs with empty `entries` sent at
        a shorter, fixed interval (typically `T/2`). You should:

        #. Cancel the previous heartbeat timer.
        #. Register a new callback that broadcasts heartbeats to followers.
        #. Use `next_index` and `match_index` to decide which log entries to
           include when followers are behind.

        Followers generally should not schedule heartbeat timers; reset the
        election timeout instead when a legitimate leader contacts them.
        """

        # TODO:
        #  * cancel old timer
        #  * schedule periodic heartbeat callback for leaders
            # Cancel old timer first
        self._cancel_heartbeat_timer()
        
        # Heartbeat interval (typically election_timeout / 2)
        # Using 50ms as a reasonable value
        heartbeat_interval_ms = 50
        
        # Schedule next heartbeat
        self._heartbeat_timer = self.scheduler.call_later(
            heartbeat_interval_ms,
            lambda: (self._send_heartbeats(), self._reset_heartbeat_timer())
        )

    def _cancel_heartbeat_timer(self) -> None:
        """
        Cancel the periodic heartbeat scheduler, if active.
        """

        # TODO:
        #  * call the stored cancel handle and clear it
        if self._heartbeat_timer is not None:
            self._heartbeat_timer()
            self._heartbeat_timer = None
