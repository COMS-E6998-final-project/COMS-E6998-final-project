"""Basic Knowledge Distillation to Train a Baseline Student

This config corresponds to step 2 in the experiment plan.

My goal here was to reflect the set up from Hinton's original paper with some default
starting values. I'm not sure if there are better default values that exist in the
pyplan, but industry standard defaults would probably be better than what I have 
here.

In terms of the student architecture, I left things super basic since I am not sure
what sort of constraints we are working with and how big we want to go.

Note: All of the properties below are complete guesses
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
		'temperature': 3, 					# They said this was good in their paper
		'alpha': .9,  						# More weight should be put on the soft labels
		'freeze_teacher': True,
	}

# Here's another way this could be set up
# @cached_property
# def loss_config(self):
# 	"""Loss function configuration."""
# 	return {
# 		# Distillation loss (soft targets from teacher)
# 		'distillation_loss': {
# 			'type': 'kl_divergence',  # KL divergence between teacher/student
# 			'weight': 0.9,
# 			'temperature': self.temperature,
# 		},
# 		# Hard label loss (ground truth)
# 		'student_loss': {
# 			'type': 'cross_entropy',
# 			'weight': 0.1,
# 		},
# 	}