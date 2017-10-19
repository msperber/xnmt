import dynet as dy

class WeightNoise(object):
  def __init__(self, std):
    self.std = std
  def __call__(self, p, train=True):
    """
    :param p: DyNet parameter (not expression)
    :param train: only apply noise if True
    :returns: DyNet expression with weight noise applied if self.std > 0
    """
    p_expr = dy.parameter(p)
    if self.std > 0.0 and train:
      p_expr = dy.noise(p_expr, self.std)
    return p_expr
    
class ResidualConnection(object):
  """
  Adds a residual connection.
  
  According to https://arxiv.org/pdf/1603.05027.pdf it is preferable to keep the residual
  connection pure, although it might be necessary to insert a linear transform to make
  layer sizes match, which can be done via the plain_resizer parameter
  """  
  def __init__(self, plain_resizer=None):
    self.plain_resizer=plain_resizer
  def __call__(self, plain, transformed):
    if self.plain_resizer:
      plain = self.plain_resizer(plain)
    return plain + transformed