import math
from collections.abc import Sequence

import numpy as np
import dynet as dy

from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.batch_norm import BatchNorm
from xnmt.nn import NiNLayer
from xnmt.serialize.serializable import Serializable
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.serialize.tree_tools import Ref, Path

class UniLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  This implements an LSTM builder based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.
  """
  yaml_tag = u'!UniLSTMSeqTransducer'
  
  def __init__(self, exp_global=Ref(Path("exp_global")), input_dim=None, hidden_dim=None,
               dropout = None, weightnoise_std=None, weight_norm = False, glorot_gain=None):
    register_handler(self)
    model = exp_global.dynet_param_collection.param_col
    input_dim = input_dim or exp_global.default_layer_dim
    hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or exp_global.dropout
    self.weightnoise_std = weightnoise_std or exp_global.weight_noise
    self.input_dim = input_dim
    
    glorot_gain = glorot_gain or exp_global.glorot_gain

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim), init=dy.GlorotInitializer(gain=glorot_gain))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=dy.GlorotInitializer(gain=glorot_gain))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))
    
    self.weight_norm = weight_norm
    if weight_norm:
      self.p_wn_wx_g = model.add_parameters(dim=(1,), init=dy.ConstInitializer(1.0))
      self.p_wn_wh_g = model.add_parameters(dim=(1,), init=dy.ConstInitializer(1.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.Wx = dy.parameter(self.p_Wx)
    self.Wh = dy.parameter(self.p_Wh)
    self.b = dy.parameter(self.p_b)
    if self.weight_norm:
      self.Wx = dy.weight_norm(self.Wx, dy.parameter(self.p_wn_wx_g))
      self.Wh = dy.weight_norm(self.Wh, dy.parameter(self.p_wn_wh_g))
    self.dropout_mask_x = None
    self.dropout_mask_h = None    

  def get_final_states(self):
    return self._final_states

  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def __call__(self, expr_seq):
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    :param expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    :returns: expression sequence
    """
    if isinstance(expr_seq, ExpressionSequence):
      expr_seq = [expr_seq]
    batch_size = expr_seq[0][0].dim()[1]
    seq_len = len(expr_seq[0])
    
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    for pos_i in range(seq_len):
      x_t = [expr_seq[j][pos_i] for j in range(len(expr_seq))]
      if isinstance(x_t, dy.Expression):
        x_t = [x_t]
      elif type(x_t) != list:
        x_t = list(x_t)
      if sum([x_t_i.dim()[0][0] for x_t_i in x_t]) != self.input_dim:
        raise ValueError("VanillaLSTMGates: x_t has inconsistent dimension %s, expecting %s" % (sum([x_t_i.dim()[0][0] for x_t_i in x_t]), self.input_dim))
      if self.dropout_rate > 0.0 and self.train:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates_t = dy.vanilla_lstm_gates_dropout_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std if self.train else 0.0)
      else:
        gates_t = dy.vanilla_lstm_gates_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.weightnoise_std if self.train else 0.0)
      c_t = dy.vanilla_lstm_c(c[-1], gates_t)
      h_t = dy.vanilla_lstm_h(c_t, gates_t)
      if expr_seq[0].mask is None or np.isclose(np.sum(expr_seq[0].mask.np_arr[:,pos_i:pos_i+1]), 0.0):
        c.append(c_t)
        h.append(h_t)
      else:
        c.append(expr_seq[0].mask.cmult_by_timestep_expr(c_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(c[-1],pos_i,False))
        h.append(expr_seq[0].mask.cmult_by_timestep_expr(h_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(h[-1],pos_i,False))
    self._final_states = [FinalTransducerState(h[-1], c[-1])]
    return ExpressionSequence(expr_list=h[1:], mask=expr_seq[0].mask)

class BiLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than the native CompactVanillaLSTMBuilder due to avoiding concat operations.
  """
  yaml_tag = u'!BiLSTMSeqTransducer'
  
  def __init__(self, exp_global=Ref(Path("exp_global")), layers=1, input_dim=None, hidden_dim=None, 
               dropout=None, weightnoise_std=None, glorot_gain=None):
    """
    :param exp_global:
    :param layers (int):
    :param input_dim (int):
    :param hidden_dim (int):
    :param dropout (float):
    :param weightnoise_std (float):
    :param glorot_gain (int or sequence of ints):
    """
    register_handler(self)
    self.num_layers = layers
    input_dim = input_dim or exp_global.default_layer_dim
    hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or exp_global.dropout
    self.weightnoise_std = weightnoise_std or exp_global.weight_noise
    assert hidden_dim % 2 == 0
    glorot_gain = glorot_gain or exp_global.glorot_gain
    self.forward_layers = [UniLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std, 
                                                glorot_gain=glorot_gain[0] if isinstance(glorot_gain, Sequence) else glorot_gain)]
    self.backward_layers = [UniLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std, 
                                                 glorot_gain=glorot_gain[0] if isinstance(glorot_gain, Sequence) else glorot_gain)]
    self.forward_layers += [UniLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std, 
                                                 glorot_gain=glorot_gain[i] if isinstance(glorot_gain, Sequence) else glorot_gain) for i in range(1, layers)]
    self.backward_layers += [UniLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std, 
                                                  glorot_gain=glorot_gain[i] if isinstance(glorot_gain, Sequence) else glorot_gain) for i in range(1, layers)]

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.last_output = []
    
  @handle_xnmt_event
  def on_collect_recent_outputs(self):
    return [(self, o) for o in self.last_output]

  def get_final_states(self):
    return self._final_states

  def __call__(self, es):
    mask = es.mask
    # first layer
    forward_es = self.forward_layers[0](es)
    rev_backward_es = self.backward_layers[0](ReversedExpressionSequence(es))
    self.last_output.append(forward_es.as_list() + rev_backward_es.as_list())

    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i]([forward_es, ReversedExpressionSequence(rev_backward_es)])
      rev_backward_es = ExpressionSequence(self.backward_layers[layer_i]([ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(), mask=mask)
      forward_es = new_forward_es
      self.last_output.append(forward_es.as_list() + rev_backward_es.as_list())

    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
                          for layer_i in range(len(self.forward_layers))]
    return ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask)
  


class CustomLSTMSeqTransducer(SeqTransducer):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.
  """
  def __init__(self, layers, input_dim, hidden_dim, exp_global=Ref(Path("exp_global")), glorot_gain=None):
    if layers!=1: raise RuntimeError("CustomLSTMSeqTransducer supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = exp_global.dynet_param_collection.param_col
    glorot_gain = glorot_gain or exp_global.glorot_gain

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim), init=dy.GlorotInitializer(gain=glorot_gain))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=dy.GlorotInitializer(gain=glorot_gain))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))

  def __call__(self, xs):
    Wx = dy.parameter(self.p_Wx)
    Wh = dy.parameter(self.p_Wh)
    b = dy.parameter(self.p_b)
    h = []
    c = []
    for i, x_t in enumerate(xs):
      if i==0:
        tmp = dy.affine_transform([b, Wx, x_t])
      else:
        tmp = dy.affine_transform([b, Wx, x_t, Wh, h[-1]])
      i_ait = dy.pick_range(tmp, 0, self.hidden_dim)
      i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pick_range(tmp, self.hidden_dim*2, self.hidden_dim*3)
      i_agt = dy.pick_range(tmp, self.hidden_dim*3, self.hidden_dim*4)
      i_it = dy.logistic(i_ait)
      i_ft = dy.logistic(i_aft + 1.0)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if i==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
      h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
    return h

class ResConvLSTMSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!ResConvLSTMSeqTransducer'
  def __init__(self, input_dim, num_filters=32, exp_global=Ref(Path("exp_global")), glorot_gain=None):
    register_handler(self)
    model = exp_global.dynet_param_collection.param_col
    if input_dim%num_filters!=0: raise RuntimeError("input_dim must be divisible by num_filters")
    self.input_dim = input_dim

    self.num_filters = num_filters
    self.freq_dim = input_dim / num_filters
    
    self.convLstm1 = ConvLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, chn_dim=num_filters, num_filters=num_filters/2, input_transposed=False, reshape_output=False, glorot_gain=glorot_gain)
    self.convLstm2 = ConvLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, chn_dim=num_filters, num_filters=num_filters/2, input_transposed=True, reshape_output=False, glorot_gain=glorot_gain)
    self.bn1 = BatchNorm(model, num_filters, 3)
    self.bn2 = BatchNorm(model, num_filters, 3)
    self.train = True
    
  def transduce(self, es):
    l1 = dy.rectify(self.bn1(self.convLstm1.transduce(es), train=self.train))
    l2 = self.bn2(self.convLstm2.transduce(l1, mask=es.mask), train=self.train)
    res = dy.rectify(es.as_tensor() + dy.transpose(dy.reshape(l2, (l2.dim()[0][0], l2.dim()[0][1]*l2.dim()[0][2]), batch_size=l2.dim()[1])))
    ret = ExpressionSequence(expr_tensor=res, mask=es.mask)
    assert len(es) == len(ret)
    return ret

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val
    
  
class ConvLSTMSeqTransducer(SeqTransducer):
  """
  This is a ConvLSTM implementation using a single bidirectional layer.
  """
  def __init__(self, input_dim, chn_dim=3, num_filters=32, input_transposed=False, reshape_output=True, exp_global=Ref(Path("exp_global")), glorot_gain=None):
    """
    :param input_dim: product of frequency and channel dimension
    :param model: DyNet parameter collection
    :param chn_dim: channel dimension
    :param input_transposed:
             True -> assume DyNet expression as input, dimensions (sent_len, freq_len, chn, batch) or (sent_len, hidden_dim, batch)
             False -> assume  ExpressionSequence of dimensions (hidden_dim, sent_len, batch) as input
    :param reshape_output:
             True -> output is an ExpressionSequence of dimensions (hidden_dim, sent_len, batch)
             False -> output is a tensor DyNet expression of dimensions (sent_len, freq, chn, batch)
    """
    model = exp_global.dynet_param_collection.param_col
    if input_dim%chn_dim!=0:
      raise RuntimeError("input_dim must be divisible by chn_dim")
    self.input_dim = input_dim

    glorot_gain = glorot_gain or exp_global.glorot_gain

    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 1
    self.filter_size_freq = 3
    initializer=dy.GlorotInitializer(gain = glorot_gain)
    self.reshape_output = reshape_output
    self.input_transposed = input_transposed
    
    self.params = {}
    for direction in ["fwd","bwd"]:
      self.params["x2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.chn_dim, self.num_filters * 4),
                               init=initializer)
      self.params["h2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.num_filters, self.num_filters * 4),
                               init=initializer)
      self.params["b_" + direction] = \
          model.add_parameters(dim=(self.num_filters * 4,), init=dy.ConstInitializer(0.0))

  def __call__(self, es, mask=None):
    if self.input_transposed:
      es_expr = es
      sent_len = es.dim()[0][0]
    else:
      mask = es.mask
      sent_len = len(es)
      es_expr = es.as_tensor()
    batch_size=es_expr.dim()[1]
    
    # if needed transpose to time-first (needed for CNN), reshape to separate channels and frequencies
    if not self.input_transposed:
      es_expr = dy.transpose(es_expr)
    es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)

    h_out = {}
    for direction in ["fwd", "bwd"]:
      # input convolutions
      gates_xt_bias = dy.conv2d_bias(es_chn, dy.parameter(self.params["x2all_" + direction]), dy.parameter(self.params["b_" + direction]), stride=(1,1), is_valid=False)
      gates_xt_bias_list = [dy.pick_range(gates_xt_bias, i, i+1) for i in range(sent_len)]

      h = []
      c = []
      for input_pos in range(sent_len):
        directional_pos = input_pos if direction=="fwd" else sent_len - input_pos - 1
        gates_t = gates_xt_bias_list[directional_pos]
        if input_pos>0:
          # recurrent convolutions
          gates_h_t = dy.conv2d(h[-1], dy.parameter(self.params["h2all_" + direction]), stride=(1,1), is_valid=False)
          gates_t += gates_h_t
        
        # standard LSTM logic
        if len(c)==0:
          c_tm1 = dy.zeros((self.freq_dim * self.num_filters,), batch_size=batch_size)
        else:
          c_tm1 = c[-1]
        # TODO: to save memory, could extend vanilla_lstm_c, vanilla_lstm_h to allow arbitrary tensors instead of just vectors; then we can avoid the reshapes below
        gates_t_reshaped = dy.reshape(gates_t, (4 * self.freq_dim * self.num_filters,), batch_size=batch_size)
        c_t = dy.reshape(dy.vanilla_lstm_c(c_tm1, gates_t_reshaped), (self.freq_dim * self.num_filters,), batch_size=batch_size) 
        h_t = dy.vanilla_lstm_h(c_t, gates_t_reshaped)
        h_t = dy.reshape(h_t, (1, self.freq_dim, self.num_filters, ), batch_size=batch_size)
        
        if mask is None or np.isclose(np.sum(mask.np_arr[:,input_pos:input_pos+1]), 0.0):
          c.append(c_t)
          h.append(h_t)
        else:
          c.append(mask.cmult_by_timestep_expr(c_t, input_pos, True) + mask.cmult_by_timestep_expr(c[-1], input_pos, False))
          h.append(mask.cmult_by_timestep_expr(h_t, input_pos, True) + mask.cmult_by_timestep_expr(h[-1], input_pos, False))

      h_out[direction] = h
    ret_expr = []
    for state_i in range(len(h_out["fwd"])):
      state_fwd = h_out["fwd"][state_i]
      state_bwd = h_out["bwd"][-1-state_i]
      if self.reshape_output:
        output_dim = (state_fwd.dim()[0][1] * state_fwd.dim()[0][2],)
      else:
        output_dim = (1, state_fwd.dim()[0][1], state_fwd.dim()[0][2],)
      fwd_reshape = dy.reshape(state_fwd, output_dim, batch_size=batch_size)
      bwd_reshape = dy.reshape(state_bwd, output_dim, batch_size=batch_size)
      ret_expr.append(dy.concatenate([fwd_reshape, bwd_reshape], d = 0 if self.reshape_output else 2))
    if self.reshape_output:
      ret = ExpressionSequence(expr_list=ret_expr, mask=mask)
    else:
      ret = dy.concatenate(ret_expr)
    return ret


class NetworkInNetworkBiLSTMTransducer(SeqTransducer, Serializable):
  """
  Builder for NiN-interleaved RNNs that delegates to regular RNNs and wires them together.
  See http://iamaaditya.github.io/2016/03/one-by-one-convolution/
  and https://arxiv.org/pdf/1610.03022.pdf
  """
  yaml_tag = u'!NetworkInNetworkBiLSTMTransducer'
  def __init__(self, layers, input_dim, hidden_dim,  
               nin_enabled=True, nin_depth=1, stride=1,
               batch_norm=False, nonlinearity="rectify", pre_activation=False, 
               weight_norm=False, weight_noise = None, dropout=None,
               exp_global=Ref(Path("exp_global")), glorot_gain_lstm=None,
               glorot_gain_nin=None):
    """
    :param exp_global:
    :param layers: depth of the network
    :param input_dim: size of the inputs of bottom layer
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param nin_enabled: whether to apply NiN units (projections (= 1x1 convolutions) + nonlinearity + batch norm units)
    :param nin_depth: number of NiN units (downsampling only performed for first projection)
    :param stride: in (first) projection layer, concatenate n frames and thus use the projection for downsampling
    :param batch_norm: uses batch norm between projection and non-linearity
    :param nonlinearity:
    :param pre_activation: True: BN -> relu -> LSTM -> [NiN -> ...] -> proj
                           False: LSTM -> [NiN -> ...]
    """
    assert layers > 0
    assert hidden_dim % 2 == 0
    assert nin_depth > 0
    register_handler(self)
    
    glorot_gain_lstm = glorot_gain_lstm or exp_global.glorot_gain
    self.builder_layers = []
    self.hidden_dim = hidden_dim
    self.stride=stride
    self.nin_depth = nin_depth
    self.nin_enabled = nin_enabled
    self.nonlinearity = nonlinearity
    self.pre_activation = pre_activation
    f = UniLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout, 
                             weight_norm=weight_norm, weightnoise_std = weight_noise,
                             glorot_gain=glorot_gain_lstm[0] if isinstance(glorot_gain_lstm, Sequence) else glorot_gain_lstm)
    b = UniLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout, 
                             weight_norm=weight_norm, weightnoise_std = weight_noise,
                             glorot_gain=glorot_gain_lstm[0] if isinstance(glorot_gain_lstm, Sequence) else glorot_gain_lstm)
    self.builder_layers.append((f, b))
    for i in range(1, layers):
      f = UniLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout, 
                              weight_norm=weight_norm, weightnoise_std = weight_noise,
                              glorot_gain=glorot_gain_lstm[i] if isinstance(glorot_gain_lstm, Sequence) else glorot_gain_lstm)
      b = UniLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout, 
                              weight_norm=weight_norm, weightnoise_std = weight_noise,
                              glorot_gain=glorot_gain_lstm[i] if isinstance(glorot_gain_lstm, Sequence) else glorot_gain_lstm)
      self.builder_layers.append((f, b))
    
    self.nin_layers = []
    if not nin_enabled:
      assert self.stride == 1
      self.nin_layers.append([]) # no pre-activation
      for i in range(layers):
        self.nin_layers.append([NiNLayer(exp_global=exp_global, input_dim=hidden_dim/2, hidden_dim=hidden_dim,
                                         use_bn=False, nonlinearity="id", use_proj=False, downsampling_factor=2,
                                         glorot_gain=glorot_gain_nin[i] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin)])
    else:
      if pre_activation:
        # first pre-activation
        self.nin_layers.append([NiNLayer(exp_global=exp_global, input_dim=input_dim, hidden_dim=input_dim,
                                         use_proj=False, use_bn=batch_norm, nonlinearity=self.nonlinearity,
                                         glorot_gain=glorot_gain_nin[0] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin)])
        for i in range(layers-1):
          nin_layer = []
          for nin_i in range(nin_depth):
            nin_layer.append(NiNLayer(exp_global=exp_global, input_dim=hidden_dim/2 if nin_i==0 else hidden_dim, hidden_dim=hidden_dim,
                                      use_bn=batch_norm, nonlinearity=self.nonlinearity, 
                                      downsampling_factor=2*self.stride if nin_i==0 else 1,
                                      glorot_gain=glorot_gain_nin[i] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin),
                             )
          self.nin_layers.append(nin_layer)
        nin_layer = []
        for nin_i in range(nin_depth-1):
          nin_layer.append(NiNLayer(exp_global=exp_global, input_dim=hidden_dim/2 if nin_i==0 else hidden_dim, hidden_dim=hidden_dim,
                                    use_bn=batch_norm, nonlinearity=self.nonlinearity, 
                                    downsampling_factor=2*self.stride if nin_i==0 else 1,
                                    glorot_gain=glorot_gain_nin[-1] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin))
        # very last layer: counterpiece to the first pre-activation
        nin_layer.append(NiNLayer(exp_global=exp_global, input_dim=hidden_dim/2 if nin_depth==1 else hidden_dim, hidden_dim=hidden_dim, 
                                  use_proj=True, use_bn=False, nonlinearity="id",
                                  downsampling_factor=2*self.stride if nin_depth==1 else 1,
                                  glorot_gain=glorot_gain_nin[-1] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin))
        self.nin_layers.append(nin_layer)
      else:
        self.nin_layers.append([]) # no pre-activation
        for i in range(layers):
          nin_layer = []
          for nin_i in range(nin_depth):
            nin_layer.append(NiNLayer(exp_global=exp_global, input_dim=hidden_dim/2 if nin_i==0 else hidden_dim, hidden_dim=hidden_dim,
                                      use_bn=batch_norm, nonlinearity=self.nonlinearity, 
                                      downsampling_factor=2*self.stride if nin_i==0 else 1,
                                      glorot_gain=glorot_gain_nin[i] if isinstance(glorot_gain_nin, Sequence) else glorot_gain_nin))
          self.nin_layers.append(nin_layer)

  @handle_xnmt_event
  def on_start_sent(self, *args, **kwargs):
    self._final_states = None
    self.last_output = []
    
  @handle_xnmt_event
  def on_collect_recent_outputs(self):
    return [(self, o) for o in self.last_output]

  def get_final_states(self):
    assert self._final_states is not None, "LSTMSeqTransducer.__call__() must be invoked before LSTMSeqTransducer.get_final_states()"
    return self._final_states
        
  def __call__(self, es):
    """
    :param es: ExpressionSequence

    """
    for nin_layer in self.nin_layers[0]:
      es = nin_layer(es)
      
    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      fs = fb(es)
      bs = bb(ReversedExpressionSequence(es))
      interleaved = []

      if es.mask is None: mask = None
      else:
        mask = es.mask.lin_subsampled(0.5) # upsample the mask to encompass interleaved fwd / bwd expressions

      for pos in range(len(fs)):
        interleaved.append(fs[pos])
        interleaved.append(bs[-pos-1])
      
      projected = ExpressionSequence(expr_list=interleaved, mask=mask)
      for nin_layer in self.nin_layers[layer_i+1]:
        projected = nin_layer(projected)
      assert math.ceil(len(es) / float(self.stride))==len(projected)
      es = projected
      if es.has_list(): self.last_output.append(es.as_list())
      else: self.last_output.append(es.as_tensor())
    
    self._final_states = [FinalTransducerState(projected[-1])]
    return projected






class QLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  This implements the quasi-recurrent neural network with input, output, and forget gate.
  https://arxiv.org/abs/1611.01576
  """
  yaml_tag = u'!QLSTMSeqTransducer'
  
  def __init__(self, exp_global=Ref(Path("exp_global")), input_dim=None, hidden_dim=None, dropout = None,
               filter_width=2, stride=1, glorot_gain=None):
    register_handler(self)
    model = exp_global.dynet_param_collection.param_col
    input_dim = input_dim or exp_global.default_layer_dim
    hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout or exp_global.dropout
    self.input_dim = input_dim
    self.stride = stride
    glorot_gain = glorot_gain or exp_global.glorot_gain

    self.p_f = model.add_parameters(dim=(filter_width, 1, input_dim, hidden_dim * 3), init=dy.GlorotInitializer(gain=glorot_gain)) # f, o, z
    self.p_b = model.add_parameters(dim=(hidden_dim * 3,), init=dy.ConstInitializer(0.0))

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, expr_seq):
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    :param expr_seq: expression sequence (will be accessed via tensor_expr
    :returns: expression sequence
    """
    
    if isinstance(expr_seq, list):
      mask_out = expr_seq[0].mask
      seq_len = len(expr_seq[0])
      batch_size = expr_seq[0].dim()[1]
      tensors = [e.as_tensor() for e in expr_seq]
      input_tensor = dy.reshape(dy.concatenate(tensors), (seq_len, 1, self.input_dim), batch_size = batch_size)
    else:
      mask_out = expr_seq.mask
      seq_len = len(expr_seq)
      batch_size = expr_seq.dim()[1]
      input_tensor = dy.reshape(dy.transpose(expr_seq.as_tensor()), (seq_len, 1, self.input_dim), batch_size = batch_size)
    
    if self.dropout > 0.0 and self.train:
      input_tensor = dy.dropout(input_tensor, self.dropout)
      
    proj_inp = dy.conv2d_bias(input_tensor, dy.parameter(self.p_f), dy.parameter(self.p_b), stride=(self.stride,1), is_valid=False)
    reduced_seq_len = proj_inp.dim()[0][0]
    proj_inp = dy.transpose(dy.reshape(proj_inp, (reduced_seq_len, self.hidden_dim*3), batch_size = batch_size))
    # proj_inp dims: (hidden, 1, seq_len), batch_size
    if self.stride > 1 and mask_out is not None:
        mask_out = mask_out.lin_subsampled(trg_len=reduced_seq_len)
    
    h = [dy.zeroes(dim=(self.hidden_dim,1), batch_size=batch_size)]
    c = [dy.zeroes(dim=(self.hidden_dim,1), batch_size=batch_size)]
    for t in range(reduced_seq_len):
      f_t = dy.logistic(dy.strided_select(proj_inp, [], [0, t], [self.hidden_dim, t+1]))
      o_t = dy.logistic(dy.strided_select(proj_inp, [], [self.hidden_dim, t], [self.hidden_dim*2, t+1]))
      z_t = dy.tanh(dy.strided_select(proj_inp, [], [self.hidden_dim*2, t], [self.hidden_dim*3, t+1]))
      
      if self.dropout > 0.0 and self.train:
        retention_rate = 1.0 - self.dropout
        dropout_mask = dy.random_bernoulli((self.hidden_dim,1), retention_rate, batch_size=batch_size)
        f_t = 1.0 - dy.cmult(dropout_mask, 1.0-f_t) # TODO: would be easy to make a zoneout dynet operation to save memory

      i_t = 1.0 - f_t
      
      if t==0:
        c_t = dy.cmult(i_t, z_t)
      else:
        c_t = dy.cmult(f_t, c[-1]) + dy.cmult(i_t, z_t)
      h_t = dy.cmult(o_t, c_t) # note: LSTM would use dy.tanh(c_t) instead of c_t
      if mask_out is None or np.isclose(np.sum(mask_out.np_arr[:,t:t+1]), 0.0):
        c.append(c_t)
        h.append(h_t)
      else:
        c.append(mask_out.cmult_by_timestep_expr(c_t,t,True) + mask_out.cmult_by_timestep_expr(c[-1],t,False))
        h.append(mask_out.cmult_by_timestep_expr(h_t,t,True) + mask_out.cmult_by_timestep_expr(h[-1],t,False))

    self._final_states = [FinalTransducerState(dy.reshape(h[-1], (self.hidden_dim,), batch_size=batch_size),\
                                               dy.reshape(c[-1], (self.hidden_dim,), batch_size=batch_size))]
    return ExpressionSequence(expr_list=h[1:], mask=mask_out)

    

class BiQLSTMSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!BiQLSTMSeqTransducer'
  
  def __init__(self, layers, input_dim=None, hidden_dim=None, dropout=None, stride=1, filter_width=2, exp_global=Ref(Path("exp_global")), glorot_gain=None):
    register_handler(self)
    self.num_layers = layers
    input_dim = input_dim or exp_global.default_layer_dim
    hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.hidden_dim = hidden_dim
    dropout = dropout or exp_global.dropout
    glorot_gain = glorot_gain or exp_global.glorot_gain
    assert hidden_dim % 2 == 0
    self.forward_layers = [QLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, stride=stride, filter_width=filter_width,
                                              glorot_gain=glorot_gain[0] if isinstance(glorot_gain, Sequence) else glorot_gain)]
    self.backward_layers = [QLSTMSeqTransducer(exp_global=exp_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, stride=stride, filter_width=filter_width,
                                               glorot_gain=glorot_gain[0] if isinstance(glorot_gain, Sequence) else glorot_gain)]
    self.forward_layers += [QLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, stride=stride, filter_width=filter_width,
                                               glorot_gain=glorot_gain[i] if isinstance(glorot_gain, Sequence) else glorot_gain) for i in range(1,layers)]
    self.backward_layers += [QLSTMSeqTransducer(exp_global=exp_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, stride=stride, filter_width=filter_width,
                                                glorot_gain=glorot_gain[i] if isinstance(glorot_gain, Sequence) else glorot_gain) for i in range(1,layers)]

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, es):
    mask = es.mask
    # first layer
    forward_es = self.forward_layers[0](es)
    rev_backward_es = self.backward_layers[0](ReversedExpressionSequence(es))

    # TODO: concat input of each layer to its output; or, maybe just add standard residual connections
    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i]([forward_es, ReversedExpressionSequence(rev_backward_es)])
      mask_out = mask
      if mask_out is not None and new_forward_es.mask.np_arr.shape != mask_out.np_arr.shape:
        mask_out = mask_out.lin_subsampled(trg_len=len(new_forward_es))
      rev_backward_es = ExpressionSequence(self.backward_layers[layer_i]([ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(), mask=mask_out)
      forward_es = new_forward_es

    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
                          for layer_i in range(len(self.forward_layers))]
    mask_out = mask
    if mask_out is not None and forward_es.mask.np_arr.shape != mask_out.np_arr.shape:
      mask_out = mask_out.lin_subsampled(trg_len=len(forward_es))
    return ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask_out)
