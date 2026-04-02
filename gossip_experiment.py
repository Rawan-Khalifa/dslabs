import random
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dslabs.simulations.sim_send_many import SimSendMany
from dslabs.nodes.node_total_order_gossip import NodeTotalOrderGossip


def run_gossip_experiment(num_nodes=8, num_messages=1000, message_delay=10,
                         num_gossip_peers=3, num_gossip_rounds=3, drop_prob=0.1,
                         num_trials=5, seed_base=1000):
    """
    Run multiple trials of gossip experiment and return statistics.
    """
    consistency_results = []
    message_counts = []

    for trial in range(num_trials):
        seed = seed_base + trial
        random.seed(seed)

        # Create simulation with gossip node
        sim = SimSendMany(
            NodeTotalOrderGossip,
            random_seed=seed,
            num_nodes=num_nodes,
            num_messages=num_messages,
            message_delay=message_delay,
            drop_prob=drop_prob,
            # Pass gossip parameters to node constructor
            node_kwargs={
                'num_gossip_peers': num_gossip_peers,
                'num_gossip_rounds': num_gossip_rounds
            }
        )

        sim.run_scenario()

        # Check consistency: all nodes should have same final value
        final_values = [node.store.get('x') for node in sim.nodes.values()]
        is_consistent = len(set(final_values)) == 1 and final_values[0] == num_messages - 1
        consistency_results.append(is_consistent)

        # Get total messages sent
        total_messages = sim.network.stats['delivered_messages'] + sim.network.stats['dropped_messages']
        message_counts.append(total_messages)

    # Calculate statistics
    consistency_rate = sum(consistency_results) / len(consistency_results)
    avg_messages = statistics.mean(message_counts)
    std_messages = statistics.stdev(message_counts) if len(message_counts) > 1 else 0

    return {
        'consistency_rate': consistency_rate,
        'avg_messages': avg_messages,
        'std_messages': std_messages,
        'consistency_results': consistency_results,
        'message_counts': message_counts
    }


def create_comprehensive_plots(results_df):
    """Create comprehensive plots from experiment results."""

    # Set up the plotting style
    plt.style.use('default')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot 1: Consistency vs Drop Probability (with error bars)
    plt.figure(figsize=(12, 8))
    drop_probs = sorted(results_df['drop_prob'].unique())

    for i, (peers, rounds) in enumerate(results_df[['peers', 'rounds']].drop_duplicates().values):
        subset = results_df[(results_df['peers'] == peers) & (results_df['rounds'] == rounds)]
        subset = subset.sort_values('drop_prob')

        consistency_rates = subset['consistency_rate'].values * 100
        consistency_stds = subset['consistency_std'].values * 100

        plt.errorbar(subset['drop_prob'], consistency_rates, yerr=consistency_stds,
                    marker='o', label=f'Peers={peers}, Rounds={rounds}',
                    color=colors[i % len(colors)], capsize=3, linewidth=2)

    plt.title('Gossip Protocol: Consistency vs Drop Probability\n(with error bars)', fontsize=14, fontweight='bold')
    plt.xlabel('Drop Probability', fontsize=12)
    plt.ylabel('Consistency Rate (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('consistency_vs_drop_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Messages vs Drop Probability (with error bars)
    plt.figure(figsize=(12, 8))

    for i, (peers, rounds) in enumerate(results_df[['peers', 'rounds']].drop_duplicates().values):
        subset = results_df[(results_df['peers'] == peers) & (results_df['rounds'] == rounds)]
        subset = subset.sort_values('drop_prob')

        plt.errorbar(subset['drop_prob'], subset['avg_messages'], yerr=subset['std_messages'],
                    marker='s', label=f'Peers={peers}, Rounds={rounds}',
                    color=colors[i % len(colors)], capsize=3, linewidth=2)

    plt.title('Gossip Protocol: Message Count vs Drop Probability\n(with error bars)', fontsize=14, fontweight='bold')
    plt.xlabel('Drop Probability', fontsize=12)
    plt.ylabel('Average Total Messages', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('messages_vs_drop_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Trade-off Scatter Plot with Size and Color Coding
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(results_df['avg_messages'], results_df['consistency_rate'] * 100,
                         c=results_df['drop_prob'], s=results_df['peers'] * 20,
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Drop Probability', fontsize=12)

    # Add legend for peer sizes
    peer_sizes = sorted(results_df['peers'].unique())
    for size in peer_sizes:
        plt.scatter([], [], c='gray', s=size * 20, label=f'Peers={size}',
                   edgecolors='black', linewidth=0.5)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Peer Count')

    plt.title('Gossip Protocol: Consistency vs Message Count Trade-off\n(Color: Drop Prob, Size: Peers)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Total Messages', fontsize=12)
    plt.ylabel('Consistency Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tradeoff_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Heatmap for each drop probability
    drop_probs = sorted(results_df['drop_prob'].unique())
    n_probs = len(drop_probs)

    fig, axes = plt.subplots(1, n_probs, figsize=(6*n_probs, 5))

    if n_probs == 1:
        axes = [axes]

    for i, drop_prob in enumerate(drop_probs):
        subset = results_df[results_df['drop_prob'] == drop_prob]

        # Create pivot table for heatmap
        pivot_consistency = subset.pivot(index='peers', columns='rounds', values='consistency_rate') * 100
        pivot_messages = subset.pivot(index='peers', columns='rounds', values='avg_messages')

        # Plot consistency heatmap
        im = axes[i].imshow(pivot_consistency, cmap='RdYlGn', aspect='auto', origin='lower')

        # Add text annotations
        for y in range(len(pivot_consistency.index)):
            for x in range(len(pivot_consistency.columns)):
                if not pd.isna(pivot_consistency.iloc[y, x]):
                    text = f'{pivot_consistency.iloc[y, x]:.1f}%'
                    axes[i].text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

        axes[i].set_title(f'Drop Prob = {drop_prob}\nConsistency (%)', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Rounds', fontsize=10)
        axes[i].set_ylabel('Peers', fontsize=10)
        axes[i].set_xticks(range(len(pivot_consistency.columns)))
        axes[i].set_yticks(range(len(pivot_consistency.index)))
        axes[i].set_xticklabels(pivot_consistency.columns)
        axes[i].set_yticklabels(pivot_consistency.index)

    plt.colorbar(im, ax=axes[-1], shrink=0.8)
    plt.tight_layout()
    plt.savefig('consistency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: Efficiency Analysis (Consistency per Message)
    plt.figure(figsize=(12, 8))

    results_df['efficiency'] = results_df['consistency_rate'] / results_df['avg_messages'] * 1000  # Scale for visibility

    for drop_prob in drop_probs:
        subset = results_df[results_df['drop_prob'] == drop_prob]
        plt.scatter(subset['avg_messages'], subset['efficiency'],
                   label=f'Drop Prob = {drop_prob}', s=50, alpha=0.7)

    plt.title('Gossip Protocol: Efficiency (Consistency per Message)\nby Drop Probability', fontsize=14, fontweight='bold')
    plt.xlabel('Average Total Messages', fontsize=12)
    plt.ylabel('Efficiency (Consistency % per 1000 Messages)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run comprehensive gossip parameter experiments."""

    # Experiment parameters
    drop_probs = [0.0, 0.1, 0.2, 0.3, 0.5]
    gossip_peers_options = [1, 2, 3, 4, 5]
    gossip_rounds_options = [1, 2, 3, 4, 5]

    print("=" * 80)
    print("GOSSIP PROTOCOL EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"Fixed parameters: num_nodes=8, num_messages=1000, message_delay=10, num_trials=5")
    print()

    # Collect all results for plotting
    all_results = []

    # Track best configurations for each drop probability
    best_configs = {}

    for drop_prob in drop_probs:
        print(f"Testing with drop_prob = {drop_prob}")
        print("-" * 50)

        best_consistency = 0
        best_config = None
        best_messages = float('inf')

        for peers in gossip_peers_options:
            for rounds in gossip_rounds_options:
                result = run_gossip_experiment(
                    drop_prob=drop_prob,
                    num_gossip_peers=peers,
                    num_gossip_rounds=rounds
                )

                consistency = result['consistency_rate']
                messages = result['avg_messages']
                std_messages = result['std_messages']

                print(f"  peers={peers}, rounds={rounds}: "
                      f"consistency={consistency:.1%}, messages={messages:.0f} ± {std_messages:.0f}")

                # Store results for plotting
                all_results.append({
                    'drop_prob': drop_prob,
                    'peers': peers,
                    'rounds': rounds,
                    'consistency_rate': consistency,
                    'consistency_std': statistics.stdev(result['consistency_results']) if len(result['consistency_results']) > 1 else 0,
                    'avg_messages': messages,
                    'std_messages': std_messages
                })

                # Track best configuration (highest consistency, then lowest messages)
                if (consistency > best_consistency or
                    (consistency == best_consistency and messages < best_messages)):
                    best_consistency = consistency
                    best_messages = messages
                    best_config = (peers, rounds, consistency, messages)

        best_configs[drop_prob] = best_config
        peers, rounds, cons, msgs = best_config
        print(f"  BEST for drop_prob={drop_prob}: peers={peers}, rounds={rounds} "
              f"(consistency={cons:.1%}, messages={msgs:.0f})")
        print()

    print("=" * 80)
    print("SUMMARY: OPTIMAL GOSSIP PARAMETERS BY DROP PROBABILITY")
    print("=" * 80)
    for drop_prob, config in best_configs.items():
        peers, rounds, cons, msgs = config
        print(f"drop_prob={drop_prob}: peers={peers}, rounds={rounds} "
              f"(consistency={cons:.1%}, messages={msgs:.0f})")

    # Create comprehensive plots
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE PLOTS...")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)
    create_comprehensive_plots(results_df)

    print("Plots saved:")
    print("  - consistency_vs_drop_detailed.png: Consistency vs drop probability with error bars")
    print("  - messages_vs_drop_detailed.png: Message count vs drop probability with error bars")
    print("  - tradeoff_scatter.png: Trade-off scatter plot (consistency vs messages)")
    print("  - consistency_heatmap.png: Heatmaps showing consistency for different peer/round combinations")
    print("  - efficiency_analysis.png: Efficiency analysis (consistency per message)")


if __name__ == "__main__":
    main()