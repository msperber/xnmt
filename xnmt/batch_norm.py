import dynet as dy
import numpy as np

class BatchNorm(object):

  bn_eps = 0.1
  bn_momentum = 0.1

  def __init__(self, model, hidden_dim, num_dim):
    self.hidden_dim = hidden_dim
    self.num_dim = num_dim
    self.bn_gamma = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(1.0))
    self.bn_beta = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(0.0))
    self.bn_population_running_mean = np.zeros((hidden_dim, ))
    self.bn_population_running_std = np.ones((hidden_dim, ))
    
  def get_normalizer_dimensionality(self):
    if self.num_dim == 1:
      return (self.hidden_dim,)
    elif self.num_dim == 2:
      return (1, self.hidden_dim,)
    elif self.num_dim == 3:
      return (1, 1, self.hidden_dim,)
    else:
      raise NotImplementedError("BatchNorm not implemented for num_dim > 3")
    
  def get_stat_dimensions(self):
    return range(self.num_dim-1)

  def bn_expr(self, input_expr, train):
    param_bn_gamma = dy.parameter(self.bn_gamma)
    param_bn_beta = dy.parameter(self.bn_beta)
    if train:
      bn_mean = dy.moment_dim(input_expr, self.get_stat_dimensions(), 1, True)
      neg_bn_mean_reshaped = -dy.reshape(-bn_mean, self.get_normalizer_dimensionality())
      self.bn_population_running_mean += -BatchNorm.bn_momentum*self.bn_population_running_mean + BatchNorm.bn_momentum * bn_mean.npvalue()
      bn_std = dy.std_dim(input_expr, self.get_stat_dimensions(), True)
      self.bn_population_running_std += -BatchNorm.bn_momentum*self.bn_population_running_std + BatchNorm.bn_momentum * bn_std.npvalue()
    else:
      neg_bn_mean_reshaped = -dy.reshape(dy.inputVector(self.bn_population_running_mean), self.get_normalizer_dimensionality())
      bn_std = dy.inputVector(self.bn_population_running_std)
    bn_numerator = input_expr + neg_bn_mean_reshaped
    bn_xhat = dy.cdiv(bn_numerator, dy.reshape(bn_std, self.get_normalizer_dimensionality()) + BatchNorm.bn_eps)
    bn_y = dy.cmult(param_bn_gamma, bn_xhat) + param_bn_beta # y = gamma * xhat + beta
    return bn_y


