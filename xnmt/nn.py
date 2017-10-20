import numpy as np
import dynet as dy
from xnmt.batcher import Mask
from xnmt.expression_sequence import ExpressionSequence
from xnmt.batch_norm import BatchNorm
from xnmt.hier_model import HierarchicalModel, recursive

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
    if plain_es.dim() != transformed_es.dim():
      raise ValueError("residual connections need matching shortcut / output dimensions, got: %s and %s" % (plain_es.dim(), transformed_es.dim()))
    return ExpressionSequence(expr_tensor=plain_es.as_tensor() + transformed_es.as_tensor(), 
                              mask=plain_es.mask, tensor_transposed=plain_es.tensor_transposed)

class TimePadder(object):
  """
  Pads ExpressionSequence along time axis.
  """
  def __init__(self, mode="zero"):
    """
    :param mode: "zero" | "repeat_last"
    """
    self.mode = mode
  def __call__(self, es, pad_len):
    """
    :param es: ExpressionSequence
    :param pad_len: how much to pad
    :returns: ExpressionSequence, with padded items indicated as masked
    """
    assert not es.tensor_transposed
    time_dim = len(es.dim()[0])-1
    single_pad_dim = list(es.dim()[0])
    single_pad_dim[time_dim] = 1
    batch_size = es.dim()[1]
    if self.mode=="zero":
      single_pad = dy.zeros(tuple(single_pad_dim), batch_size=batch_size)
    elif self.mode=="repeat_last":
      single_pad = es[-1]
    mask = es.mask
    if mask is not None:
      mask_dim = (mask.np_arr.shape[0], pad_len)
      mask = Mask(np.append(mask.np_arr, np.ones(mask_dim), axis=1))
    if es.has_list():
      es_list = es.as_list()
      es_list.extend([single_pad] * pad_len)
      return ExpressionSequence(expr_list=es_list, mask=mask)
    else:
      raise NotImplementedError("tensor padding not implemented yet")
      

class NiNLayer(HierarchicalModel):
  def __init__(self, yaml_context, input_dim, hidden_dim, use_proj=True,
               use_bn=True, nonlinearity="relu", downsampling_factor=1):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.use_proj = use_proj
    self.use_bn = use_bn
    self.nonlinearity = nonlinearity
    self.downsampling_factor = downsampling_factor
    self.timePadder = TimePadder(mode="zero")
    if downsampling_factor < 1: raise ValueError("downsampling_factor must be >= 1")
    if not use_proj:
      if hidden_dim!=input_dim*downsampling_factor: raise ValueError("disabling projections requires hidden_dim == input_dim*downsampling_factor") 
    if use_proj:
      self.p_proj = yaml_context.dynet_param_collection.param_col.add_parameters(dim=(hidden_dim, input_dim*downsampling_factor))
    if self.use_bn:
      self.bn = BatchNorm(yaml_context.dynet_param_collection.param_col, hidden_dim, 2, time_first=False)
      
  def __call__(self, es):
    """
    :param es: ExpressionSequence of dimensions input_dim x time
    :returns: ExpressionSequence
              if use_proj: dimensions = hidden x ceil(time/downsampling_factor)
              else:        dimensions = (input_dim*downsampling_factor) x ceil(time/downsampling_factor)
    """
    assert not es.tensor_transposed
    if not es.dim()[0][0] == self.input_dim:
      raise ValueError("This NiN Layer requires inputs of hidden dim %s, got %s." % (self.input_dim, es.dim()[0][0]))

    if self.use_proj:
      if len(es) % self.downsampling_factor!=0:
        es = self.timePadder(es, pad_len = self.downsampling_factor - (len(es) % self.downsampling_factor))

    if es.mask is None: mask_out = None
    else:
      if self.downsampling_factor==1:
        mask_out = es.mask
      else:
        mask_out = es.mask.lin_subsampled(self.downsampling_factor)

    projections = []
    expr_list = es.as_list()
    for pos in range(0, len(expr_list), self.downsampling_factor):
      if self.downsampling_factor > 1:
        concat = dy.concatenate(expr_list[pos : pos+self.downsampling_factor])
      else:
        # TODO: in case of no downsampling, we could put time into batch dimension and compute all matrix multiplies in parallel
        concat = expr_list[pos]
        
      if self.use_proj:
        proj = dy.parameter(self.p_proj)
        proj = proj * concat
      else: proj = concat
      projections.append(proj)

    if self.use_bn:
      bn_layer = self.bn(dy.concatenate(projections, 1), 
                         train=self.train,
                         mask=mask_out)
      nonlin = self.apply_nonlinearity(bn_layer, self.nonlinearity)
      return ExpressionSequence(expr_tensor=nonlin, mask=mask_out)
    else:
      es = []
      for proj in projections:
        nonlin = self.apply_nonlinearity(proj, self.nonlinearity)
        es.append(nonlin)
      return ExpressionSequence(expr_list=es, mask=mask_out)

  def apply_nonlinearity(self, expr, nonlinearity):
    if nonlinearity is None:
      return expr
    elif nonlinearity.lower()=="relu":
      return dy.rectify(expr)
    else:
      raise RuntimeError("unknown nonlinearity %s" % nonlinearity)
  
  @recursive
  def set_train(self, val):
    self.train = val