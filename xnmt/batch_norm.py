import dynet as dy
import numpy as np

class BatchNorm(object):

  bn_eps = 0.1
  bn_momentum = 0.1

  def __init__(self, model, hidden_dim, num_dim, time_first=True):
    self.hidden_dim = hidden_dim
    self.num_dim = num_dim
    self.bn_gamma = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(1.0))
    self.bn_beta = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(0.0))
    self.bn_population_running_mean = np.zeros((hidden_dim, ))
    self.bn_population_running_std = np.ones((hidden_dim, ))
    self.time_first = time_first
    if not self.time_first: raise RuntimeError("time_first=False may be broken & needs to be fixed")
    
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

  def __call__(self, input_expr, train, mask=None):
    """
    :param input_expr:
    :param train: if True, compute batch statistics, if False, use precomputed statistics
    :param mask: compute statistics only over unmasked parts of the input expression
    :param time_first: if False, assume input_expr[hidden..][time][batch]; if True, assume input_expr[time][hidden..][batch] 
    """
    dim_in = input_expr.dim()
    param_bn_gamma = dy.parameter(self.bn_gamma)
    param_bn_beta = dy.parameter(self.bn_beta)
    if train:
      num_unmasked = 0
      if mask is not None:
        input_expr = mask.set_masked_to_mean(input_expr, self.time_first)
        num_unmasked = (mask.np_arr.size - np.count_nonzero(mask.np_arr)) * mask.broadcast_factor(input_expr)
      bn_mean = dy.moment_dim(input_expr, self.get_stat_dimensions(), 1, True, num_unmasked)
      neg_bn_mean_reshaped = -dy.reshape(-bn_mean, self.get_normalizer_dimensionality())
      self.bn_population_running_mean += (-BatchNorm.bn_momentum)*self.bn_population_running_mean + BatchNorm.bn_momentum * bn_mean.npvalue()
      bn_std = dy.std_dim(input_expr, self.get_stat_dimensions(), True, num_unmasked)
      self.bn_population_running_std += (-BatchNorm.bn_momentum)*self.bn_population_running_std + BatchNorm.bn_momentum * bn_std.npvalue()
    else:
      neg_bn_mean_reshaped = -dy.reshape(dy.inputVector(self.bn_population_running_mean), self.get_normalizer_dimensionality())
      bn_std = dy.inputVector(self.bn_population_running_std)
    bn_numerator = input_expr + neg_bn_mean_reshaped
    bn_xhat = dy.cdiv(bn_numerator, dy.reshape(bn_std, self.get_normalizer_dimensionality()) + BatchNorm.bn_eps)
    bn_y = dy.cmult(param_bn_gamma, bn_xhat) + param_bn_beta # y = gamma * xhat + beta
    dim_out = bn_y.dim()
    assert dim_out == dim_in
    return bn_y


