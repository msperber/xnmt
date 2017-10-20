import dynet as dy
from xnmt.expression_sequence import ExpressionSequence

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
  
  According to https://arxiv.org/pdf/1603.05027.pdf it is preferable to keep the shortcut
  connection pure (i.e., None), although it might be necessary to insert a linear transform to make
  layer sizes match, which can be done via the plain_resizer parameter
  (see advice here: https://github.com/fchollet/keras/issues/2608 )
  """  
  def __init__(self, shortcut_operation=None):
    self.shortcut_operation = shortcut_operation
  def __call__(self, plain_es, transformed_es):
    if self.shortcut_operation:
      plain_es = self.shortcut_operation(plain_es)
    return ExpressionSequence(expr_tensor=plain_es.as_tensor() + transformed_es.as_tensor(), 
                              mask=plain_es.mask, tensor_transposed=plain_es.tensor_transposed)

class Padder(object):
  TIME_DIMENSION = -1
  BATCH_DIMENSION = 2398343
  def __init__(self, mode="zero"):
    """
    :param mode: "zero" | "repeat_last"
    """
    self.mode = mode
  def __call__(self, expr, dim, pad_len):
    if self.mode=="zero":
      # TODO: something like
      zero_pad = dy.inputTensor(np.zeros(es[0].dim()[0]+(es[0].dim()[1],)), batched=True)
      es.extend([zero_pad] * (stride-len(es)%stride))
    elif self.mode=="repeat_last":
      pass # TODO

class NiNLayer(object):
  def __init__(self, projection=True, batch_norm=True, nonlinearity="relu", stride=1):
    self.projection = projection
    self.use_bn = batch_norm
    self.nonlinearity = nonlinearity
    self.stride = stride
    self.padder = Padder(mode="zero")
    if stride < 1: raise ValueError("stride must be >= 1")
    if not projection and stride > 1: raise ValueError("striding requires projections enabled")
    if projection:
      
  def __call__(self, es):
    assert not es.tensor_transposed
    expr_list = es.as_list()
    if self.projection:
      if len(es) % self.stride!=0:
        expr_list = self.padder(expr_list, dim=self.padder.TIME_DIMENSION, 
                                pad_len = self.stride - (len(es) % self.stride))


  def apply_one_nin(self, es, bn, stride, lintransf, downsampled_mask=None):
    batch_size = es[0].dim()[1]
    if len(es)%stride!=0:
      
      # TODO: could pad by replicating last timestep instead
      zero_pad = dy.inputTensor(np.zeros(es[0].dim()[0]+(es[0].dim()[1],)), batched=True)
      es.extend([zero_pad] * (stride-len(es)%stride))
    projections = []
    lintransf_param = dy.parameter(lintransf)
    # TODO: could speed this up by putting time steps into the batch dimension and thereby avoiding the for loop
    for pos in range(0, len(es), stride):
      concat = dy.concatenate(es[pos:pos+stride])
      if self.projection_enabled:
        proj = lintransf_param * concat
      else: proj = concat
      projections.append(proj)
    if self.use_bn:
      bn_layer = bn(dy.concatenate([dy.reshape(x, (1,self.hidden_dim), batch_size=batch_size) for x in projections], 
                                0), 
                 train=self.train,
                 mask=downsampled_mask)
      nonlin = self.apply_nonlinearity(bn_layer)
      es = [dy.pick(nonlin, i) for i in range(nonlin.dim()[0][0])]
    else:
      es = []
      for proj in projections:
        nonlin = self.apply_nonlinearity(proj)
        es.append(nonlin)
    return es
      
      