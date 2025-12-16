"""Basic Knowledge Distillation to Train a Baseline Student

This config corresponds to step 2 in the experiment plan.

My goal here was to reflect the set up from Hinton's original paper with some default
starting values. I'm not sure if there are better default values that exist in the
pyplan, but industry standard defaults would probably be better than what I have 
here.

In terms of the student architecture, I left things super basic since I am not sure
what sort of constraints we are working with and how big we want to go.
"""

@cached_property
def student_model(self):
	"""Student architecture"""
	return {
		'architecture': {
			'hidden_layers': [256, 128, 64],
			'activation': 'relu',
			'use_batch_norm': True,
			'dropout_rate': 0.2,
		},
  		'learning_parameters': {
			'learning_rate_scale': self.learning_rate_scale,
   			'l1_regularization_strength': 0.0,
          	'l2_regularization_strength': 0.0,
			'optimizer': learning_parameters_pb2.LearningParameters.ADAMW,
			'min_value': self.embed_range.min,
			'max_value': self.embed_range.max,
			'initialization': {
				'constant': self.embed_init_constant,
				'accum_initial_value': self.accum_initial_value,
			},
	},
	}

@cached_property
def distillation_params(self):
	"""Core KD parameters"""
	return {
		'method': 'hard_and_soft_targets',
		'teacher': "path/to/teacher",
		'temperature': [1.5, 2, 3, 4, 6], 	        # sweep over different temperatures
		'alpha': [.3, .5, .7, .9],		        # weight on soft targets vs. hard labels 
		'freeze_teacher': True,

		# Adding this for a FitNet-style intermediate loss
		'intermediate_losses': [{
			# Specify which layers to match
			'name': 'hint_1',
			'student_layer_idx': 1,           	# Should be halfway through student
			'teacher_layer_idx': -2,          	# Should be halfway through teacher
			'loss': 'mse',

			# Need to handle dimension mismatch - use a linear projection
			'projection': {
				'type': 'linear',
				'out_dim': 128,
				'bias': False,
			},

			# weight (beta) — how much to weight this intermediate loss
			'weight': [0.05, 0.1, 0.2],		# sweep over different weights

			# Scheduler to ramp down the intermediate loss over time (less hinting)
			'schedule': 'cosine decay'
		}],
	}
