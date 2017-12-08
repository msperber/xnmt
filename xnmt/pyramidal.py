from __future__ import division, generators

import numpy as np
import dynet as dy
from xnmt.batcher import Mask
from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState



class PyramidalLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  Builder for pyramidal RNNs that delegates to regular RNNs and wires them together.
  See https://arxiv.org/abs/1508.01211

  Every layer (except the first) reduces sequence length by factor 2.

      builder = PyramidalRNNBuilder(4, 128, 100, model, VanillaLSTMBuilder)
      [o1,o2,o3] = builder.transduce([i1,i2,i3])
  """
  yaml_tag = u'!PyramidalLSTMSeqTransducer'
  
  def __init__(self, yaml_context, layers=1, input_dim=None, hidden_dim=None,
               downsampling_method="concat", reduce_factor=2, dropout=None):
    """
    :param layers: depth of the PyramidalRNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    :param downsampling_method: how to perform downsampling (concat|skip|random-skip)
    :param reduce_factor: integer, or list of ints (different skip for each layer)
    """
    register_handler(self)
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    input_dim = input_dim or yaml_context.default_layer_dim
    self.dropout = dropout or yaml_context.dropout
    assert layers > 0
    assert hidden_dim % 2 == 0
    assert downsampling_method in ["concat", "skip", "random-skip"]
    assert type(reduce_factor)==int or (type(reduce_factor)==list and len(reduce_factor)==layers-1)
    self.builder_layers = []
    self.downsampling_method = downsampling_method
    self.reduce_factor = reduce_factor
    self.input_dim = input_dim
    f = UniLSTMSeqTransducer(yaml_context, input_dim, hidden_dim / 2, dropout=dropout)
    b = UniLSTMSeqTransducer(yaml_context, input_dim, hidden_dim / 2, dropout=dropout)
    self.builder_layers.append((f, b))
    for _ in range(layers - 1):
      layer_input_dim = hidden_dim if downsampling_method in ["skip","random-skip"] else hidden_dim*reduce_factor
      f = UniLSTMSeqTransducer(yaml_context, layer_input_dim, hidden_dim / 2, dropout=dropout)
      b = UniLSTMSeqTransducer(yaml_context, layer_input_dim, hidden_dim / 2, dropout=dropout)
      self.builder_layers.append((f, b))

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def _reduce_factor_for_layer(self, layer_i):
    if layer_i >= len(self.builder_layers)-1:
      return 1
    elif type(self.reduce_factor)==int:
      return self.reduce_factor
    else:
      return self.reduce_factor[layer_i]

  def __call__(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs,
    and concatenating.

    :param es: an ExpressionSequence
    """

    es_list = [es]

    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      reduce_factor = self._reduce_factor_for_layer(layer_i)
      
      if self.downsampling_method == "random-skip":
        downsample_sel = np.nonzero(np.random.binomial(1, 1.0 / reduce_factor, len(es_list[0])))[0]
        
      if es_list[0].mask is None: mask_out = None
      else:
        if self.downsampling_method == "random-skip":
          mask_out = Mask(es_list[0].mask.np_arr[:,downsample_sel])
        else:
          mask_out = es_list[0].mask.lin_subsampled(reduce_factor)
      
      if self.downsampling_method=="concat" and len(es_list[0]) % reduce_factor != 0:
        raise ValueError("For 'concat' subsampling, sequence lengths must be multiples of the total reduce factor. Configure batcher accordingly.")
      fs = fb(es_list)
      bs = bb([ReversedExpressionSequence(es_item) for es_item in es_list])
      if layer_i < len(self.builder_layers) - 1:
        if self.downsampling_method=="skip":
          es_list = [ExpressionSequence(expr_list=fs[::reduce_factor], mask=mask_out), ExpressionSequence(expr_list=bs[::reduce_factor][::-1], mask=mask_out)]
        elif self.downsampling_method == "random-skip":
          es_list = [ExpressionSequence(expr_list=[fs[i] for i in downsample_sel], mask=mask_out), ExpressionSequence(expr_list=[bs[i] for i in downsample_sel][::-1], mask=mask_out)]
        elif self.downsampling_method=="concat":
          es_len = len(es_list[0])
          es_list_fwd = []
          es_list_bwd = []
          for i in range(0, es_len, reduce_factor):
            for j in range(reduce_factor):
              if i==0:
                es_list_fwd.append([])
                es_list_bwd.append([])
              es_list_fwd[j].append(fs[i+j])
              es_list_bwd[j].append(bs[len(es_list[0])-reduce_factor+j-i])
          es_list = [ExpressionSequence(expr_list=es_list_fwd[j], mask=mask_out) for j in range(reduce_factor)] + [ExpressionSequence(expr_list=es_list_bwd[j], mask=mask_out) for j in range(reduce_factor)]
        else:
          raise RuntimeError("unknown downsampling_method %s" % self.downsampling_method)
      else:
        # concat final outputs
        ret_es = ExpressionSequence(expr_list=[dy.concatenate([f, b]) for f, b in zip(fs, ReversedExpressionSequence(bs))], mask=mask_out)
    

    self._final_states = [FinalTransducerState(dy.concatenate([fb.get_final_states()[0].main_expr(),
                                                            bb.get_final_states()[0].main_expr()]),
                                            dy.concatenate([fb.get_final_states()[0].cell_expr(),
                                                            bb.get_final_states()[0].cell_expr()])) \
                          for (fb, bb) in self.builder_layers]

    return ret_es
