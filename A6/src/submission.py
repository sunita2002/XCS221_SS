#!/usr/bin/python
"""
XCS221 Homework 6: Bayesian Networks
"""

from ctypes import util
import ctypes.util
from ctypes import util

from util import (
    BayesianNetwork, BayesianNode, init_zero_conditional_probability_tables,
    normalize_counts, load_annotation_csv, plot_annotator_cpts, plot_label_cpt
)
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import sys
import os

# Ensure local directory is prioritized for module resolution
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import util
import numpy as np
from typing import List, Dict, Tuple, Optional

############################################################
# Problem 2a: Converting Phylogenetic Tree to Bayesian Network

############################################################
# Problem 2a: Converting Phylogenetic Tree to Bayesian Network

def initialize_phylogenetic_tree(mutation_rate: float, genome_length: int = 1) -> BayesianNetwork:
    """
    Initialize the phylogenetic tree as a Bayesian network using the BayesianNode and
    BayesianNetwork classes in util.py.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")
    
    # ### START CODE HERE ###
    nucleotides = ['A', 'C', 'T', 'G']
    
    # Root node: Thomas bayus (shape must be genome_length x 4)
    thomas_prior = np.full((genome_length, 4), 0.25, dtype=float)
    thomas_bayus = BayesianNode(
        name='Thomas bayus',
        parents=[],
        domain=nucleotides,
        #conditional_prob_table=thomas_prior
    )
    # Then set the desired CPT
    #thomas_bayus.cpt = thomas_bayus   # numpy array of the right shape

    # Mutation CPT shape: (4, 4)
    mutation_cpt = np.eye(4) * (1.0 - mutation_rate) + (1.0 - np.eye(4)) * (mutation_rate / 3.0)
    
    # 2. Humblus studentus (Child of Thomas bayus)
    humblus_studentus = BayesianNode(
        name='Humblus studentus',
        parents=[thomas_bayus],
        domain=nucleotides,
        conditional_prob_table=mutation_cpt.copy()
    )
    
    # 3. Aryamus bayus (Child of Thomas bayus)
    aryamus_bayus = BayesianNode(
        name='Aryamus bayus',
        parents=[thomas_bayus],
        domain=nucleotides,
        conditional_prob_table=mutation_cpt.copy()
    )
    
    # 4. Kenius bayus (Child of Aryamus bayus)
    kenius_bayus = BayesianNode(
        name='Kenius bayus',
        parents=[aryamus_bayus],
        domain=nucleotides,
        conditional_prob_table=mutation_cpt.copy()
    )

    network = BayesianNetwork([aryamus_bayus, humblus_studentus, thomas_bayus, kenius_bayus], batch_size=genome_length)
    return network

    # ### END CODE HERE ###
    network = BayesianNetwork([aryamus_bayus, humblus_studentus, thomas_bayus, kenius_bayus], batch_size=genome_length)
    return network

############################################################
# Problem 2b: Sampling from Bayesian Networks

def forward_sampling(network: BayesianNetwork) -> Dict[str, str]:
    """
    Sample a single observation from the given Bayesian network.

    Use the topological ordering of variables in network.order and sample each
    variable according to its conditional probability distribution given the values
    of its parents (which have already been sampled).

    Args:
        network: A BayesianNetwork object containing nodes to sample from

    Returns:
        A dictionary mapping variable names to their sampled values
    """
    samples: Dict[str, List[str]] = {node.name: [] for node in network.order}

    for idx in range(network.batch_size):
        assignment: Dict[str, str] = {}
        
        # ### START CODE HERE ###
############################################################
# Problem 2b: Sampling from Bayesian Networks

def forward_sampling(network: BayesianNetwork) -> Dict[str, List[str]]:
    """
    Sample a single observation from the given Bayesian network.

    Use the topological ordering of variables in network.order and sample each
    variable according to its conditional probability distribution given the values
    of its parents (which have already been sampled).

    Args:
        network: A BayesianNetwork object containing nodes to sample from

    Returns:
        A dictionary mapping variable names to their sampled values
    """
    domain = ['A', 'C', 'T', 'G']
    val_to_idx = {val: i for i, val in enumerate(domain)}
    order = network.order if hasattr(network, 'order') else network.nodes
    samples: Dict[str, List[str]] = {node.name: [] for node in order}

    for idx in range(network.batch_size):
        assignment: Dict[str, str] = {}
        
        # ### START CODE HERE ###
        for node in order:
            if not node.parents:
                if hasattr(node, 'conditional_prob_table') and isinstance(node.conditional_prob_table, np.ndarray):
                    probs = node.conditional_prob_table[idx, :]
                else:
                    probs = [0.25, 0.25, 0.25, 0.25]
            else:
                parent_vals = [assignment[p.name if hasattr(p, 'name') else p] for p in node.parents]
                parent_indices = tuple(val_to_idx[pv] for pv in parent_vals)
                
                if hasattr(node, 'conditional_prob_table') and isinstance(node.conditional_prob_table, np.ndarray):
                    if len(parent_indices) == 1:
                        probs = node.conditional_prob_table[parent_indices[0], :]
                    else:
                        probs = node.conditional_prob_table[parent_indices]
                elif hasattr(node, 'cpt') and node.cpt is not None:
                    parent_values = parent_vals[0] if len(parent_vals) == 1 else tuple(parent_vals)
                    probs = []
                    for val in domain:
                        key = parent_values + (val,) if node.parents else val
                        if key not in node.cpt:
                            key = (val,) if not node.parents else parent_values + (val,)
                        probs.append(node.cpt[key])
                else:
                    probs = [0.25, 0.25, 0.25, 0.25]
                
            sampled_val = np.random.choice(domain, p=probs)
            assignment[node.name] = sampled_val
            samples[node.name].append(sampled_val)
            
        # ### END CODE HERE ###

    return samples


############################################################
# Problem 2c: Computing Joint Probability

def compute_joint_probability(
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    batch_indices: Optional[List[int]] = None
) -> float:
    """
    Compute the joint probability of a given assignment to all variables in the network.

    Args:
        network: A BayesianNetwork object
        assignment: Dictionary mapping variable names to their assigned values
        batch_indices: Optional list of batch indices to compute the joint probability for
            (default is [0, 1, ..., batch_size-1], so all the indices)

    Returns:
        The joint probability as a float
    """
    
    # ### START CODE HERE ###
# Problem 2c: Computing Joint Probability

def compute_joint_probability(
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    batch_indices: Optional[List[int]] = None
) -> float:
    """
    Compute the joint probability of a given assignment to all variables in the network.

    Args:
        network: A BayesianNetwork object
        assignment: Dictionary mapping variable names to their assigned values
        batch_indices: Optional list of batch indices to compute the joint probability for
            (default is [0, 1, ..., batch_size-1], so all the indices)

    Returns:
        The joint probability as a float
    """
    # ### START CODE HERE ###
    if batch_indices is None:
        batch_indices = list(range(network.batch_size))

    domain = ['A', 'C', 'T', 'G']
    val_to_idx = {val: i for i, val in enumerate(domain)}
    order = network.order if hasattr(network, 'order') else network.nodes
    total_prob = 1.0

    for idx in batch_indices:
        sample_prob = 1.0
        for node in order:
            name = getattr(node, 'name', getattr(node, 'variable', None))
            val = assignment[name][idx]
            val_idx = val_to_idx[val]

            if not node.parents:
                if hasattr(node, 'conditional_prob_table') and isinstance(node.conditional_prob_table, np.ndarray):
                    sample_prob *= node.conditional_prob_table[idx, val_idx]
                else:
                    sample_prob *= node.cpt.get(val, 0.25)
            else:
                parent_vals = [assignment[p.name if hasattr(p, 'name') else p][idx] for p in node.parents]
                parent_indices = tuple(val_to_idx[pv] for pv in parent_vals)
                
                if hasattr(node, 'conditional_prob_table') and isinstance(node.conditional_prob_table, np.ndarray):
                    if len(parent_indices) == 1:
                        sample_prob *= node.conditional_prob_table[parent_indices[0], val_idx]
                    else:
                        idx_tuple = parent_indices + (val_idx,)
                        sample_prob *= node.conditional_prob_table[idx_tuple]
                else:
                    key = parent_vals[0] if len(parent_vals) == 1 else tuple(parent_vals)
                    key = key + (val,) if isinstance(key, tuple) else (key, val)
                    sample_prob *= node.cpt.get(key, 1.0)
                    
        total_prob *= sample_prob

    return total_prob

    # ### END CODE HERE ###

# ############################################################
# # Problem 2d: Test forward sampling

def test_forward_sampling():
    np.random.seed(123)
    random.seed(123)
    
    # ### START CODE HERE ###
    network = initialize_phylogenetic_tree(mutation_rate=0.1, genome_length=10)
    sample = forward_sampling(network)
    joint_probability = compute_joint_probability(network, sample)

    # ### END CODE HERE ###
    #print(sample)
    #print(f"{joint_probability:.10%}")

# Uncomment to test forward sampling
test_forward_sampling()

############################################################
# Problem 2e: Rejection Sampling

def rejection_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_samples: int
) -> Dict[Any, float]:
    """
    Use rejection sampling to estimate the likelihoods for each outcome in the 
    target variable conditioned on the given assignments (a dictionary mapping
    variable names to their assigned values), i.e. P(target_variable | conditioned_on_assignments).

    Args:
        network: A BayesianNetwork object
        target_variable: The name of the variable to estimate the likelihoods for
        conditioned_on_assignments: A dictionary mapping variable names to their assigned values
        num_samples: The number of samples to draw

    Returns:
        A dictionary mapping outcomes to their likelihoods, conditioned on the given assignments.
    """
    counts = {}
    total_accepted = 0

    # ### START CODE HERE ###
    for _ in range(num_samples):
        sample = forward_sampling(network)
        
        # Check if the sample matches all evidence in conditioned_on_assignments
        match = True
        for var, assigned_vals in conditioned_on_assignments.items():
            if sample.get(var) != assigned_vals:
                match = False
                break
                
        if match:
            total_accepted += 1
            target_val = sample[target_variable]
            
            # Ensure the outcome is hashable (convert lists to tuples)
            if isinstance(target_val, list):
                outcome = tuple(target_val)
            else:
                outcome = target_val
                
            counts[outcome] = counts.get(outcome, 0) + 1

    # Normalize counts to obtain likelihoods / probabilities
    probabilities = {}
    if total_accepted > 0:
        for outcome, count in counts.items():
            probabilities[outcome] = count / total_accepted
   

    return probabilities
    # ### END CODE HERE ###

############################################################
# Problem 2f: Gibbs Sampling

def gibbs_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_iterations: int,
    initial_state: Dict[str, List[str]] = None
) -> Dict[Any, float]:
    """
    Estimate P(target_variable | conditioned_on_assignments) via Gibbs sampling.

    - Initialization: random ancestral sample (forward_sampling)
    - Use compute_joint_probability to score single-variable proposals
      (simple and robust for small networks)
    - Record the Thomas bayus genome after each full sweep
    """
    # random initialization, then set evidence variables
    state = forward_sampling(network) if initial_state is None else initial_state
    for evidence_var, evidence_val in conditioned_on_assignments.items():
        state[evidence_var] = evidence_val

    # resample all non-evidence vars each sweep
    resample_nodes = [n for n in network.order if n.name not in conditioned_on_assignments]
    counts = defaultdict(int)

    for _ in range(num_iterations):
        
        # ### START CODE HERE ###
        for node in resample_nodes:
            for idx in range(network.batch_size):
                domain = node.domain
                weights = []
                for val in domain:
                    old_val = state[node.name][idx]
                    state[node.name][idx] = val
                    weights.append(compute_joint_probability(network, state))
                    state[node.name][idx] = old_val
                
                total = sum(weights)
                probs = [w / total for w in weights] if total > 0 else [1.0 / len(domain)] * len(domain)
                sampled_val = np.random.choice(domain, p=probs)
                state[node.name][idx] = sampled_val

        # Find the correct key for target_variable in state (handling case/spacing differences)
        target_key = target_variable
        if target_key not in state:
            for k in state.keys():
                if k.lower().replace(" ", "") == target_variable.lower().replace(" ", ""):
                    target_key = k
                    break

        target_val = state[target_key]
        outcome = tuple(target_val) if isinstance(target_val, list) else target_val
        counts[outcome] += 1

        # ### END CODE HERE ###
    total_samples = sum(counts.values())
    return {val: counts[val] / total_samples for val in counts.keys()}

def test_gibbs_vs_rejection(
    num_steps: int=10000,
    seed: int=0,
    mutation_rate: float=0.1,
    genome_length: int=4,
):
    def exact_inference():
        same = 1.0 - mutation_rate
        diff = mutation_rate / 3.0
        p_a = same * same + 3 * diff * diff
        p_not_a = diff * same + same * diff + 2 * diff * diff
        expected = (p_a ** 3) * p_not_a
        return expected
    np.random.seed(seed)
    random.seed(seed)
    network = initialize_phylogenetic_tree(mutation_rate=mutation_rate, genome_length=genome_length)
    kenius_bayus_genome = ['A'] * genome_length

    vals = gibbs_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_iterations=num_steps)
    gibbs_p_aaac = vals.get(('A', 'A', 'A', 'C'), 0.0)

    rejection_vals = rejection_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_samples=num_steps)
    rejection_p_aaac = rejection_vals.get(('A', 'A', 'A', 'C'), 0.0)

    #print(f"{'Gibbs':>10} {gibbs_p_aaac:.4f}")
    #print(f"{'Rejection':>10} {rejection_p_aaac:.4f}")
    #print(f"{'Exact':>10} {exact_inference():.4f}")

# Uncomment to test Gibbs vs. rejection sampling
test_gibbs_vs_rejection(num_steps=100, mutation_rate=0.1, genome_length=4)
# test_gibbs_vs_rejection(num_steps=10000, mutation_rate=0.1, genome_length=4)

 
############################################################
# Problem 3d: Bayesian Network for Annotators

def bayesian_network_for_annotators(num_annotators: int, dataset_size: int=1) -> BayesianNetwork:
    """
    Return the Bayesian network for the annotators.
    """
    # ### START CODE HERE ###
    domain = ['good', 'bad']
    
    # 1. Latent ground-truth node Y with shape (dataset_size, 2) and uniform prior (0.5)
    #y_cpt = np.full((dataset_size, 2), 0.5, dtype=float)
    y_cpt = np.full((1, 2), 0.5, dtype=float)
    y_node = BayesianNode(
        name='Y',
        parents=[],
        domain=domain,
        conditional_prob_table=y_cpt
    )
    
    nodes = [y_node]
    #print(f'nodes:{nodes}')
    # 2. Annotator CPT: Diagonally dominant representing better-than-chance accuracy
    # Row 0: P(A | Y='good') -> [P(good | good), P(bad | good)] = [0.8, 0.2]
    # Row 1: P(A | Y='bad')  -> [P(good | bad), P(bad | bad)] = [0.2, 0.8]
    annotator_cpt = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    
    for i in range(num_annotators):
        annotator_node = BayesianNode(
            name=f"A_{i}",
            parents=[y_node],
            domain=domain,
            conditional_prob_table=annotator_cpt.copy()
        )
        nodes.append(annotator_node)
        
    network = BayesianNetwork(nodes, batch_size=dataset_size)
    return network
    # ### END CODE HERE ###

############################################################
# Problem 3e: Maximum likelihood estimation


def accumulate_assignment(
    counts: Dict[str, np.ndarray],
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    weight: float = 1.0,
    batch_indices: Optional[List[int]] = None,
) -> None:
    """
    Add weighted counts for a fully or partially observed assignment.
    """
    batch_size = len(list(assignment.values())[0])
    for i in range(batch_size) if batch_indices is None else range(len(batch_indices)):
        idx = batch_indices[i] if batch_indices is not None else i
        assignment_i = {k: v[i] for k, v in assignment.items()}
        for node in network.nodes:
# ### START CODE HERE ###
            if node.name not in assignment_i:
                continue
            if any(p.name not in assignment_i for p in node.parents):
                continue
            
            val_idx = node.domain.index(assignment_i[node.name])
            if not node.parents:
                counts[node.name][idx, val_idx] += weight
            else:
                parent_indices = tuple(p.domain.index(assignment_i[p.name]) for p in node.parents)
                counts[node.name][parent_indices + (val_idx,)] += weight
            # ### END CODE HERE ###


def mle_estimation(network: BayesianNetwork, data: List[Dict[str, List[str]]], lambda_param: float = 1.0) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE.
    """
# ### START CODE HERE ###
    counts = util.init_zero_conditional_probability_tables(network)
    for assignment in data:
        accumulate_assignment(counts, network, assignment)
    
    # normalize_counts modifies the network parameters in-place[cite: 3]
    #util.normalize_counts(network, counts)

     # Update each node's CPT using counts with Laplace smoothing
    for node in network.nodes:
        node_counts = counts[node.name]
        smoothed_counts = node_counts + lambda_param
        denominator = np.sum(smoothed_counts, axis=-1, keepdims=True)
        node.conditional_prob_table = smoothed_counts / denominator


    return network
# ### END CODE HERE ###

############################################################
# Problem 3f: Maximum likelihood estimation for annotators

def mle_estimation_for_annotators(data: List[Dict[str, List[str]]]) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE for the annotators.
    """
# ### START CODE HERE ###
    # Determine the number of annotators from the keys (any key starting with 'A_')
    annotator_keys = [k for k in data[0].keys() if k.startswith('A_')]
    num_annotators = len(annotator_keys)
    
    # Combine data chunks into a single batched assignment dictionary
    combined_data = {k: [] for k in data[0].keys()}
    for d in data:
        for k, v in d.items():
            combined_data[k].extend(v)
            
    dataset_size = len(combined_data['Y'])
    
    # Initialize the Bayesian network for the annotators with the correct size
    network = bayesian_network_for_annotators(num_annotators, dataset_size)
    
    # Estimate parameters using MLE (with default Laplace smoothing lambda_param=1.0)
    trained_network = mle_estimation(network, [combined_data], lambda_param=1.0)
    
    return trained_network
    # ### END CODE HERE ###

def test_mle_estimation_for_annotators():
    data = load_annotation_csv('data/annotations.csv', include_labels=True)
    trained = mle_estimation_for_annotators(data)
    plot_annotator_cpts(trained, "plots/annotators.png")

#test_mle_estimation_for_annotators()

if __name__ == '__main__':
    test_mle_estimation_for_annotators()
############################################################

# Problem 4a: Expectation step

def e_step(
    network: BayesianNetwork, data: List[Dict[str, List[str]]]
) -> Tuple[List[Dict[str, List[str]]], List[float], List[List[int]]]:
    """
    Create the dataset of fully-observed weighted observations given some hidden variables, for the EM algorithm.
    """
    # ### START CODE HERE ###
    import itertools

    completions = []
    weights = []
    indices = []
    
    for data_idx, assignment in enumerate(data):
        batch_size = len(list(assignment.values())[0])
        for b_idx in range(batch_size):
            obs_i = {k: v[b_idx] for k, v in assignment.items()}
            
            hidden_nodes = [node for node in network.nodes if node.name not in obs_i]
            
            if not hidden_nodes:
                comp = {k: [v] for k, v in obs_i.items()}
                completions.append(comp)
                weights.append(1.0)
                indices.append([data_idx])
            else:
                hidden_names = [node.name for node in hidden_nodes]
                hidden_domains = [node.domain for node in hidden_nodes]
                
                raw_comps = []
                joint_probs = []
                
                for combo in itertools.product(*hidden_domains):
                    comp_i = dict(obs_i)
                    for h_name, val in zip(hidden_names, combo):
                        comp_i[h_name] = val
                    
                    # Compute joint probability directly from network CPTs
                    jp = 1.0
                    for node in network.nodes:
                        val = comp_i[node.name]
                        val_idx = node.domain.index(val)
                        if not node.parents:
                            # Root node Y has shape (dataset_size, 2)
                            jp *= node.conditional_prob_table[b_idx, val_idx]
                        else:
                            # Conditional node (annotator) with parent indices
                            parent_indices = tuple(p.domain.index(comp_i[p.name]) for p in node.parents)
                            jp *= node.conditional_prob_table[parent_indices + (val_idx,)]
                    
                    batched_comp_i = {k: [v] for k, v in comp_i.items()}
                    raw_comps.append(batched_comp_i)
                    joint_probs.append(jp)
                
                total_jp = sum(joint_probs)
                if total_jp == 0:
                    norm_weights = [1.0 / len(joint_probs)] * len(joint_probs)
                else:
                    norm_weights = [jp / total_jp for jp in joint_probs]
                
                for comp, wt in zip(raw_comps, norm_weights):
                    completions.append(comp)
                    weights.append(wt)
                    indices.append([data_idx])
                    
    return completions, weights, indices
    # ### END CODE HERE ###

############################################################
# Problem 4b: Maximization step

def m_step(
    network: BayesianNetwork,
    all_completions: List[Dict[str, List[str]]],
    all_weights: List[float],
    all_indices: List[List[int]],
) -> BayesianNetwork:
    """
    Update the CPTs of the Bayesian network using expected counts.
    """
    # ### START CODE HERE ###
    counts = util.init_zero_conditional_probability_tables(network)
    
    for comp, weight, indices in zip(all_completions, all_weights, all_indices):
        accumulate_assignment(counts, network, comp, weight=weight, batch_indices=indices)
        
    # normalize_counts updates the network parameters in-place
    util.normalize_counts(network, counts)
            
    return network
    # ### END CODE HERE ###


############################################################
# Problem 4c: Expectation maximization

def em_learn(network: BayesianNetwork, data: List[Dict[str, List[str]]], num_iterations: int) -> BayesianNetwork:
    """
    Run the EM algorithm for a given number of iterations.
    """
    # ### START CODE HERE ###
    for _ in range(num_iterations):
        # 1. E-step: compute expected completions and soft weights given current parameters
        completions, weights, indices = e_step(network, data)
        
        # 2. M-step: update network parameters using the expected counts from completions
        network = m_step(network, completions, weights, indices)
        
    return network
    # ### END CODE HERE ###

def test_em_learn():
    network = bayesian_network_for_annotators(num_annotators=3, dataset_size=100)
    data = load_annotation_csv('data/annotations.csv', include_labels=False)
    trained = em_learn(network, data, num_iterations=100)
    plot_annotator_cpts(trained, "plots/annotators_em.png")
    plot_label_cpt(trained, "plots/labels_em.png")

# Uncomment to test EM learning
test_em_learn()

def test_em_learn():
    network = bayesian_network_for_annotators(num_annotators=3, dataset_size=100)
    data = load_annotation_csv('data/annotations.csv', include_labels=False)
    trained = em_learn(network, data, num_iterations=100)
    plot_annotator_cpts(trained, "plots/annotators_em.png")
    plot_label_cpt(trained, "plots/labels_em.png")

# Uncomment to test EM learning
test_em_learn()
