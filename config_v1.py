# Author: Rohan Singh
# Code for a simple version 1 config to be imported to train the pyplan model
# This is a no-op change 

# this is an example of a property that we might want to change
@cached_property
def embed_learning(self):
  """Embedding parameters."""
  return {
      'learning_parameters': {
          'learning_rate_scale': self.learning_rate_scale,
          'l1_regularization_strength': 0.0,
          'l2_regularization_strength': 0.0,
          'optimizer': learning_parameters_pb2.LearningParameters.ADAGRAD,
          'min_value': self.embed_range.min,
          'max_value': self.embed_range.max,
          'initialization': {
              'constant': self.embed_init_constant,
              'accum_initial_value': self.accum_initial_value,
          },
      },
      'missing_value_is_zero': self.force_missing_value_is_zero,
  }
  