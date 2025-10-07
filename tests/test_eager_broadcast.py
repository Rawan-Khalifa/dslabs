import random

import pytest

from dslabs.simulations.sim_send_many import SimSendMany
from dslabs.nodes.node_eager_broadcast import NodeEagerBroadcast


def _run_eager_broadcast(seed: int, **kwargs):
    random.seed(seed)
    sim = SimSendMany(NodeEagerBroadcast, random_seed=seed, **kwargs)
    sim.run_scenario()
    return sim


def test_eager_broadcast_converges_without_drops_when_spaced_out():
    """
    With no drops and a large inter-message delay, all nodes should converge to
    the last written value for key 'x'. This serves as a baseline passing test.
    """
    num_messages = 10
    sim = _run_eager_broadcast(
        seed=7,
        num_nodes=5,
        num_messages=num_messages,
        message_delay=1000,  # >> max network delay; preserves global order
        drop_prob=0.0,
    )
    values = sim.results["stored_values"]
    assert set(values.values()) == {num_messages - 1}


def test_eager_broadcast_handles_drops_on_last_write_replication():
    """
    With eager broadcasting, even with a high drop probability, the redundant 
    broadcasts should ensure all nodes eventually receive all updates.
    This test should PASS with eager broadcasting.
    """
    num_messages = 12
    sim = _run_eager_broadcast(
        seed=12345,
        num_nodes=8,
        num_messages=num_messages,
        message_delay=1000,
        drop_prob=0.5,  # high drop rate
    )
    values = sim.results["stored_values"]
    # With eager broadcasting, everyone should have the final value
    assert set(values.values()) == {num_messages - 1}


@pytest.mark.xfail
def test_eager_broadcast_still_fails_due_to_reordering_with_fast_client_rate():
    """
    Even with eager broadcasting, a fast client rate plus variable network delay 
    can still cause older writes to arrive after newer ones, leaving some nodes 
    with stale final values. Eager broadcasting helps with drops but doesn't 
    solve ordering issues.
    """
    num_messages = 20
    sim = _run_eager_broadcast(
        seed=42,
        num_nodes=6,
        num_messages=num_messages,
        message_delay=10,  # << network delay range; invites reordering
        drop_prob=0.0,  # we are _not_ dropping messages
    )
    values = sim.results["stored_values"]
    # This should still fail because eager broadcasting doesn't solve ordering
    assert set(values.values()) == {num_messages - 1}
