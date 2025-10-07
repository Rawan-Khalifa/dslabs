import random

import pytest

from dslabs.simulations.sim_send_many import SimSendMany
from dslabs.nodes.node_ordered_delivery import NodeOrderedDelivery


def _run_ordered_delivery(seed: int, **kwargs):
    random.seed(seed)
    sim = SimSendMany(NodeOrderedDelivery, random_seed=seed, **kwargs)
    sim.run_scenario()
    return sim


def test_ordered_delivery_converges_without_drops_when_spaced_out():
    """
    With no drops and a large inter-message delay, all nodes should converge to
    the last written value for key 'x'. This serves as a baseline passing test.
    """
    num_messages = 10
    sim = _run_ordered_delivery(
        seed=7,
        num_nodes=5,
        num_messages=num_messages,
        message_delay=1000,  # >> max network delay; preserves global order
        drop_prob=0.0,
    )
    values = sim.results["stored_values"]
    
    # Check that all nodes have the same final value
    assert set(values.values()) == {num_messages - 1}
    
    # Check that all nodes have the same log (ordered delivery)
    logs = {nid: node.log for nid, node in sim.nodes.items()}
    first_log = next(iter(logs.values()))
    for nid, log in logs.items():
        assert log == first_log, f"Node {nid} has different log: {log} vs {first_log}"


def test_ordered_delivery_handles_drops():
    """
    With ordered delivery, even with message drops, all nodes should converge 
    to the same final value. Logs may differ slightly due to drops, but should
    be very similar.
    """
    num_messages = 12
    sim = _run_ordered_delivery(
        seed=12345,
        num_nodes=8,
        num_messages=num_messages,
        message_delay=1000,
        drop_prob=0.5,  # high drop rate
    )
    values = sim.results["stored_values"]
    
    # Check that all nodes have the same final value
    assert set(values.values()) == {num_messages - 1}
    
    # Check that logs are mostly similar (allow for some differences due to drops)
    logs = {nid: node.log for nid, node in sim.nodes.items()}
    log_lengths = [len(log) for log in logs.values()]
    
    # Most nodes should have delivered most messages
    min_length = min(log_lengths)
    max_length = max(log_lengths)
    assert min_length >= num_messages - 3, f"Some nodes missing too many messages: {log_lengths}"
    assert max_length <= num_messages, f"Some nodes have too many messages: {log_lengths}"
    
    # All nodes should end with the same final value in their logs
    final_log_values = [log[-1][1] if log else None for log in logs.values()]
    assert set(final_log_values) == {num_messages - 1}, f"Different final log values: {final_log_values}"


def test_ordered_delivery_handles_reordering_with_fast_client_rate():
    """
    With ordered delivery, even with fast client rate that causes reordering,
    all nodes should converge and have the same ordered log.
    """
    num_messages = 20
    sim = _run_ordered_delivery(
        seed=42,
        num_nodes=6,
        num_messages=num_messages,
        message_delay=10,  # << network delay range; invites reordering
        drop_prob=0.0,  # we are _not_ dropping messages
    )
    values = sim.results["stored_values"]
    
    # Check that all nodes have the same final value
    assert set(values.values()) == {num_messages - 1}
    
    # Check that all nodes have the same log (ordered delivery)
    logs = {nid: node.log for nid, node in sim.nodes.items()}
    first_log = next(iter(logs.values()))
    for nid, log in logs.items():
        assert log == first_log, f"Node {nid} has different log: {log} vs {first_log}"


def test_ordered_delivery_handles_both_drops_and_reordering():
    """
    The ultimate test: ordered delivery should handle both drops and reordering.
    """
    num_messages = 15
    sim = _run_ordered_delivery(
        seed=999,
        num_nodes=6,
        num_messages=num_messages,
        message_delay=10,  # Fast rate causes reordering
        drop_prob=0.3,     # Some drops
    )
    values = sim.results["stored_values"]
    
    # Check that all nodes have the same final value
    assert set(values.values()) == {num_messages - 1}
    
    # Check that all nodes have the same log (ordered delivery)
    logs = {nid: node.log for nid, node in sim.nodes.items()}
    first_log = next(iter(logs.values()))
    for nid, log in logs.items():
        assert log == first_log, f"Node {nid} has different log: {log} vs {first_log}"
