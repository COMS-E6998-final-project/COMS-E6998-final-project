"""
Post-training pruning methods

Idea here is based on this paper that proposes a framework for doing pruning
without needing to retrain/fine tune after

"Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining"
        Paper: https://arxiv.org/html/2508.15828v1
        Repo: https://github.com/sazzadadib/Z-Pruner/tree/main
        Core idea: use row/col normalized z-scores to highlight outliers and identify 
        redundant parameters
"""

@cached_property
def z_prune(self):
        return {
                "method": "z_score_importance",
                "apply_after_training": True,
                "layer_wise_execution": True,
                "sparsity_ratio": [.1, .2, .3, .4, .5], # percent of weights pruned
                "pruning_mode": ["global", "neuron"], # prune across entire weight matrix or per neuron
                "weight_normalization": { # normalize weights before computing their importance
                        "row_wise_l2": True,
                        "column_wise_l2": True,
                        "epsilon": 1e-8,
                },
                "z_score": {
                        "compute_per_layer": True, # compute z scores layer by layer
                        "cubic_amplification": True, # apply cubic amplification by doing z^3
                        "amplification_power": [2.0, 3.0, 4.0], # sweep amplification power as a test
                },
                "layer_importance_multipliers": { # make some layers more or less likely to be deemed important
                        "attention_layers": 1.0, # standard
                        "ffn_layers": [.8, .9, 1.0], # probably more prunable
                        "embedding_layers": 1.2 # less prunable
                },

                ## run forward pass (no backprop, no weight updates) and collect info to use in importance scoring
                "activation_measurement": {
                        "num_samples": [64, 128, 256],
                        "sequence_length": 512,
                        "batch_size": 8,
                        "data_source": "training",
                        "collect_statistics": {
                                "input_activations": True,
                                "output_activations": True,
                                "activation_norms": True
                        },
                        "aggregation": "mean",
                        "requires_grad": False,
                        "inference_only": True,
                },
        }