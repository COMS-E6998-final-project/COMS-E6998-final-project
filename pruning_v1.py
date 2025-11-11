"""
Pruning ideas from lecture 

Basic process (sensitivity analysis): for each layer i in the model, prune with ratio {0, .1, .2, ..., .9}
and observe accuracy after pruning. Pick degradation threshold for accuracy to get overall desired pruning ratio
--> does not account for interaction between layers, expect to be less optimal

NetAdapt: for each iteration, prune each layer by manually defined amount, short term fine tune (10k iterations), 
then prune layer with highest accuracy. Repeat until achieve overall desired pruning ratio. 

After pruning (both cases above), need to fine tune to recover accuracy -- use lr 1/10 - 1/100 of original lr

Regularization: add to loss term to penalize nonzero params + encourage smaller params. 
Most common for improving pruning performance are L1 & L2

pytorch pruning example: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/f40ae04715cdb214ecba048c12f8dddf/pruning_tutorial.ipynb

TODO: look into sparsity optimized tensor core specifics for our hardware (pufferfish TPUs)

"""
@cached_property
def prune(self):
	"""Pruning basic configuration"""
	return {
		"method": ["sensitivity_analysis", "netadapt"],  ## the two methods from lecture we could try 
		'learning_parameters': {
   			'l1_regularization_strength': [1e-3, 1e-4, 1e-5],  ## ideally can experiment w these values as well
          		'l2_regularization_strength': [1e-3, 1e-4, 1e-5],  ## ideally can experiment w these values as well
		},
		"sensitivity_analysis_parameters": {
			"pruning_scope": "layer",  ## or might be called "local"
			"accuracy_degradation_threshold": [.02, .05, .08],  ## TODO change these depending on current accuracy
			"layer_pruning_sweep_ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  ## sweep per layer according to notes above
		},
		"netadapt_parameters": {
			"pruning_scope": "global", 
			"pruning_step_sweep": [0.01, 0.05, 0.1],  ## how much to prune per iteration (overall, not per layer) 
			"short_term_finetune_iterations": 10000,  ## num finetuning iterations per pruning iteration
			"stopping_criteria": {  ## not sure how to phrase this, but idea is we stop iterating upon any of these conditions
				"max_iterations": 100,  ## or some other cutoff value
				"target_sparsity": .9,  ## if we've hit overall 90% sparsity, since seems like almost always downhill after that
				"min_accuracy": .88,  ## if our accuracy has dropped below 88% (change this depending on initial acc)
			},
		},
		"post_pruning_finetune": {
			"num_iterations": 100000,  ## TODO check reasonable value for this
			"learning_rate": .0001,  ## or "learning_rate_scale": [.1, .01] --> should be 1/10-1/100 of original lr 
		}
	}
  