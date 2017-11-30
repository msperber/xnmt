import math
import dynet as dy
from xnmt.batch_norm import BatchNorm
from xnmt.expression_sequence import ExpressionSequence
from xnmt.nn import WeightNoise
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import Transducer, SeqTransducer
from xnmt.serializer import Serializable


class StridedConvSeqTransducer(SeqTransducer, Serializable):
  """
  Implements several (possibly strided) CNN layers. No padding is performed, thus layer size will shrink even with striding turned off.
  """
  yaml_tag = u'!StridedConvSeqTransducer'
    
  def __init__(self, yaml_context, init_gauss_var=0.1, weight_noise=0.0,
               layers=1, input_dim=120, chn_dim=3, num_filters=32, stride=(2,2), 
               batch_norm=False, nonlinearity=None, pre_activation=False,
               output_tensor=False, transpose=True):
    """
    :param param_col:
    :param init_gauss_var: initialize filter weights with Gaussian noise of given variance
    :param weight_noise: apply Gaussian noise of given standard deviation to weights (training time only)
    :param layers: encoder depth
    :param input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim 
    :param chn_dim: channel input dimension
    :param num_filters: channel output dimension
    :param stride: tuple, downsample via striding
    :param batch_norm: apply batch normalization before the nonlinearity. Normalization is performed over batch, time, and frequency dimensions (and not over the channel dimension).
    :param nonlinearity: e.g. "rectify" / "silu" / ... / "maxout" / None (note: "maxout" will double number of filter parameters, but leave output dimension unchanged)
    :param pre_activation: If True, please BN + nonlinearity before CNN
    :param output_transposed_tensor: True -> output is a expression sequence holding a 3d-tensor (including channel dimension), in transposed form (time is first dimension)
                                     False -> output is a expression sequence holding a list of flat vector expressions (frequency and channel dimensions are merged)
    """
    register_handler(self)
    assert layers > 0
    if input_dim % chn_dim != 0:
      raise ValueError("StridedConvEncoder requires input_dim mod chn_dim == 0, got: %s and %s" % (input_dim, chn_dim))
    
    param_col = yaml_context.dynet_param_collection.param_col
    self.layers = layers
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.stride = stride
    self.output_transposed_tensor = output_tensor
    self.nonlinearity = nonlinearity or yaml_context.nonlinearity
    self.pre_activation = pre_activation
    
    self.use_bn = batch_norm
    self.train = True
    self.transpose = transpose
    self.weight_noise = WeightNoise(weight_noise)
    
    normalInit=dy.NormalInitializer(0, init_gauss_var)
    self.bn_layers = []
    self.filters_layers = []
    for layer_i in range(layers):
      filters = param_col.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters * (2 if self.nonlinearity=="maxout" else 1)),
                                     init=normalInit)
      if self.use_bn:
        self.bn_layers.append(BatchNorm(param_col, (self.chn_dim if self.pre_activation else self.num_filters) * (2 if self.nonlinearity=="maxout" else 1), 3))
      self.filters_layers.append(filters)
  
  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def get_output_dim(self):
    conv_dim = self.freq_dim
    for layer_i in range(self.layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_freq + 1) / float(self.get_stride_for_layer(layer_i)[1])))
    return conv_dim * self.num_filters
  
  def get_stride_for_layer(self, layer_i):
    if type(self.stride)==tuple: return self.stride
    else:
      assert type(self.stride)==list
      return self.stride[layer_i]
      
  
  def get_output_len(self, input_len):
    conv_dim = input_len
    for layer_i in range(self.layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_time + 1) / float(self.get_stride_for_layer(layer_i)[0])))
    return conv_dim

  def pad(self, expr, pad_size):
    assert pad_size>=0
    if pad_size==0:
      return expr
    return dy.concatenate([expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=expr.dim()[1])]) # TODO: replicate last frame instead of padding zeros

  def apply_nonlinearity(self, nonlinearity, expr):
    if nonlinearity=="rectify":
      return dy.rectify(expr)
    if nonlinearity=="silu":
      return dy.silu(expr)
    elif nonlinearity=="maxout":
      raise NotImplementedError("maxout not yet implemented")
      # TODO:
      # - split channels into 2 groups
      # - return dy.bmax(cnn_layer_group1, cnn_layer_group2)
    elif nonlinearity is not None:
      raise RuntimeError("unknown nonlinearity: %s" % nonlinearity)
    return expr
    
  def __call__(self, es):
    es_expr = es.as_tensor()
    if not es.tensor_transposed:
      es_expr = dy.transpose(es_expr, [1,0])

    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    es_expr = self.pad(es_expr, pad_size)
    sent_len += pad_size

    # loop over layers
    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn
    mask_out = None
    for layer_i in range(len(self.filters_layers)):
      cnn_filter = self.weight_noise(self.filters_layers[layer_i], self.train)

      if not self.pre_activation:
        cnn_layer = dy.conv2d(cnn_layer, cnn_filter, stride=self.get_stride_for_layer(layer_i), is_valid=True)

      if self.use_bn:
        mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
        cnn_layer = self.bn_layers[layer_i](cnn_layer, train=self.train, mask=mask_out)
          
      cnn_layer = self.apply_nonlinearity(self.nonlinearity, cnn_layer)

      if self.pre_activation:
        cnn_layer = dy.conv2d(cnn_layer, cnn_filter, stride=self.get_stride_for_layer(layer_i), is_valid=True)
      
    mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
    if self.output_transposed_tensor:
      return ExpressionSequence(expr_tensor=cnn_layer, mask=mask_out, tensor_transposed=True)
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1]*cnn_layer.dim()[0][2]), batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return ExpressionSequence(expr_list=es_list, mask=mask_out)


class PoolingConvSeqTransducer(SeqTransducer, Serializable):
  """
  Implements several CNN layers, with strided max pooling interspersed.
  """
  yaml_tag = u'!PoolingConvSeqTransducer'
  
  def __init__(self, yaml_context, input_dim, pooling=[None, (1,1)], chn_dim=3, num_filters=32, 
               output_tensor=False, nonlinearity=None, init_gauss_var=0.1):
    """
    :param layers: encoder depth
    :param input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim 
    :param model
    :param chn_dim: channel dimension
    :param num_filters
    :param output_tensor: if set, the output is directly given as a 3d-tensor, rather than converted to a list of vector expressions
    :param nonlinearity: "rely" / "maxout" / None
    """
    raise Exception("TODO: buggy, needs proper transposing")
    assert input_dim % chn_dim == 0
    
    model = yaml_context.dynet_param_collection.param_col
    self.layers = len(pooling)
    assert self.layers > 0
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.pooling = pooling
    self.output_tensor = output_tensor
    self.nonlinearity = nonlinearity or yaml_context.nonlinearity
    
    normalInit=dy.NormalInitializer(0, init_gauss_var)
    self.bn_layers = []
    self.filters_layers = []
    self.bn_alt_layers = []
    self.filters_alt_layers = []
    for layer_i in range(self.layers):
      filters = model.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters),
                                     init=normalInit)
      self.filters_layers.append(filters)
      if self.nonlinearity=="maxout":
        filters_alt = model.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters),
                                     init=normalInit)
        self.filters_alt_layers.append(filters_alt)
  
  def get_output_dim(self):
    conv_dim = self.freq_dim
    for layer_i in range(self.layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_freq + 1) / float(self.get_stride_for_layer(layer_i)[1])))
    return conv_dim * self.num_filters
  
  def get_stride_for_layer(self, layer_i):
    if self.pooling[layer_i]:
      return self.pooling[layer_i]
    else:
      return (1,1) 
  
  def get_output_len(self, input_len):
    conv_dim = input_len
    for layer_i in range(self.layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_time + 1) / float(self.get_stride_for_layer(layer_i)[0])))
    return conv_dim

  def __call__(self, es):
    es_expr = es.as_tensor()

    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    if pad_size>0:
      es_expr = dy.concatenate([es_expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=es_expr.dim()[1])])
      sent_len += pad_size

    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn
    
    # loop over layers
    for layer_i in range(len(self.filters_layers)):
      cnn_layer_prev = cnn_layer
      filters = self.filters_layers[layer_i]
      
      # convolution
      cnn_layer = dy.conv2d(cnn_layer, dy.parameter(filters), stride=(1,1), is_valid=True)
      
      # non-linearity
      if self.nonlinearity=="maxout":
        filters_alt = self.filters_alt_layers[layer_i]
        cnn_layer_alt = dy.conv2d(cnn_layer_prev, dy.parameter(filters_alt), stride=(1,1), is_valid=True)
      if self.nonlinearity=="rectify":
        cnn_layer = dy.rectify(cnn_layer)
      elif self.nonlinearity=="silu":
        cnn_layer = dy.silu(cnn_layer)
      elif self.nonlinearity=="maxout":
        cnn_layer = dy.bmax(cnn_layer, cnn_layer_alt)
      elif self.nonlinearity is not None:
        raise RuntimeError("unknown nonlinearity: %s" % self.nonlinearity)
      
      # max pooling
      if self.pooling[layer_i]:
        cnn_layer = dy.maxpooling2d(cnn_layer, (3,3), stride=self.pooling[layer_i], is_valid=True)
      
    mask_out = es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
    if self.output_tensor:
      return ExpressionSequence(tensor_expr=cnn_layer, mask=mask_out)
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1]*cnn_layer.dim()[0][2]), batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return ExpressionSequence(list_expr=es_list, mask=mask_out)

class ConvStrideTransducer(Transducer):
  def __init__(self, chn_dim, stride=(1,1), margin=(0,0)):
    self.chn_dim = chn_dim
    self.stride = stride
    self.margin = margin
  def __call__(self, expr):
    return dy.strided_select(expr, [self.margin[0], expr.dim()[0][0]-self.margin[0], self.stride[0],
                                    self.margin[1], expr.dim()[0][1]-self.margin[1], self.stride[1],
                                    0,              expr.dim()[0][2],                  1,
                                    0,              expr.dim()[1],                     1])
