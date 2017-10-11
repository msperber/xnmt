import math
import dynet as dy
from xnmt.batch_norm import BatchNorm
from xnmt.expression_sequence import ExpressionSequence

class StridedConvEncBuilder(object):
  """
  Implements several strided CNN layers.
  """
    
  def __init__(self, layers, input_dim, model, chn_dim=3, num_filters=32, 
               output_tensor=False, batch_norm=False, stride=(2,2), nonlinearity="relu",
               init_gauss_var=0.1, transpose=True):
    """
    :param layers: encoder depth
    :param input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim 
    :param model
    :param chn_dim: channel dimension
    :param num_filters
    :param output_tensor: if set, the output is directly given as a 3d-tensor, rather than converted to a list of vector expressions
    :param batch_norm:
    :param stride:
    :param nonlinearity: "rely" / "maxout" / None
    :param transpose: indicates that inputs will be given in feat x time format and need to be transposed
    """
    assert layers > 0
    assert input_dim % chn_dim == 0
    
    self.layers = layers
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.stride = stride
    self.output_tensor = output_tensor
    self.nonlinearity = nonlinearity
    
    self.use_bn = batch_norm
    self.train = True
    self.transpose = transpose
    
    normalInit=dy.NormalInitializer(0, init_gauss_var)
    self.bn_layers = []
    self.filters_layers = []
    self.bn_alt_layers = []
    self.filters_alt_layers = []
    for layer_i in range(layers):
      filters = model.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters),
                                     init=normalInit)
      if nonlinearity=="maxout":
        filters_alt = model.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters),
                                     init=normalInit)
      if self.use_bn:
        self.bn_layers.append(BatchNorm(model, self.num_filters, 3))
        if nonlinearity=="maxout":
          self.bn_alt_layers.append(BatchNorm(model, self.num_filters, 3))
      self.filters_layers.append(filters)
      if nonlinearity=="maxout": self.filters_alt_layers.append(filters_alt)
  
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

  def whoami(self): return "StridedConvEncBuilder"

  def set_dropout(self, p):
    if p>0.0: raise NotImplementedError("StridedConvEncBuilder does not support dropout")
  def disable_dropout(self):
    pass

  def transduce(self, es):
    es_expr = es.as_tensor()
    if self.transpose:
      es_expr = dy.transpose(es_expr, [1,0]) # TODO: to save memory, we could implement ExpressionSequence(transpose=True)

    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    if pad_size>0:
      es_expr = dy.concatenate([es_expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=es_expr.dim()[1])])
      sent_len += pad_size

    # loop over layers
    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn
    mask_out = None
    for layer_i in range(len(self.filters_layers)):
      cnn_layer_prev = cnn_layer
      filters = self.filters_layers[layer_i]
      cnn_layer = dy.conv2d(cnn_layer, dy.parameter(filters), stride=self.get_stride_for_layer(layer_i), is_valid=True)
      if self.nonlinearity=="maxout":
        filters_alt = self.filters_alt_layers[layer_i]
        cnn_layer_alt = dy.conv2d(cnn_layer_prev, dy.parameter(filters_alt), stride=self.get_stride_for_layer(layer_i), is_valid=True)
      if self.use_bn:
        mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
        cnn_layer = self.bn_layers[layer_i].bn_expr(cnn_layer, train=self.train, mask=mask_out, is_transposed=True)
        if self.nonlinearity=="maxout":
          cnn_layer_alt = self.bn_alt_layers[layer_i].bn_expr(cnn_layer_alt, train=self.train)
      if self.nonlinearity=="relu":
        cnn_layer = dy.rectify(cnn_layer)
      elif self.nonlinearity=="maxout":
        cnn_layer = dy.bmax(cnn_layer, cnn_layer_alt)
      elif self.nonlinearity is not None:
        raise RuntimeError("unknown nonlinearity: %s" % self.nonlinearity)
    mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
    if self.output_tensor:
      return ExpressionSequence(expr_tensor=cnn_layer, mask=mask_out)
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1]*cnn_layer.dim()[0][2]), batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return ExpressionSequence(expr_list=es_list, mask=mask_out)



class PoolingConvEncBuilder(object):
  # TODO: buggy, needs proper transposing
  """
  Implements several CNN layers, with strided max pooling interspersed.
  """
  
  def __init__(self, input_dim, model, pooling=[None, (1,1)], chn_dim=3, num_filters=32, 
               output_tensor=False, nonlinearity="relu", init_gauss_var=0.1):
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
    assert input_dim % chn_dim == 0
    
    self.layers = len(pooling)
    assert self.layers > 0
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.pooling = pooling
    self.output_tensor = output_tensor
    self.nonlinearity = nonlinearity
    
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
      if nonlinearity=="maxout":
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

  def whoami(self): return "PoolingConvEncBuilder"

  def set_dropout(self, p):
    if p>0.0: raise NotImplementedError("PoolingConvEncBuilder does not support dropout")
  def disable_dropout(self):
    pass

  def transduce(self, es):
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
      if self.nonlinearity=="relu":
        cnn_layer = dy.rectify(cnn_layer)
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
