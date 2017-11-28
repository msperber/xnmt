import dynet as dy
import math
import numpy as np
from xnmt.expression_sequence import ExpressionSequence
from xnmt.nn import LayerNorm, Linear, PositionwiseFeedForward, TimeDistributed, PositionwiseLinear
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event


MAX_SIZE = 5000
MIN_VAL = -10000   # This value is close to NEG INFINITY


class MultiHeadedAttention(object):
  def __init__(self, head_count, model_dim, model, downsample_factor=1, input_dim=None, 
               is_self_att=False, ignore_masks=False, broadcast_masks=False, plot_attention=None):
    if input_dim is None: input_dim = model_dim
    self.input_dim = input_dim
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count
    assert downsample_factor >= 1
    self.downsample_factor = downsample_factor
    self.plot_attention = plot_attention
    self.plot_attention_counter = 0
    
    self.ignore_masks = ignore_masks
    self.broadcast_masks = broadcast_masks
    
    self.is_self_att = is_self_att
    
    if is_self_att:
      self.linear_kvq = Linear(input_dim, head_count * self.dim_per_head * 3, model)
      
    else:
      self.linear_keys = Linear(input_dim, head_count * self.dim_per_head, model)
      self.linear_values = Linear(input_dim, head_count * self.dim_per_head, model)
      self.linear_query = Linear(input_dim, head_count * self.dim_per_head, model)

    # Layer Norm Module
    self.layer_norm = LayerNorm(model_dim, model)
    
    if model_dim!=input_dim: self.res_shortcut = PositionwiseLinear(input_dim, model_dim, model)

  def __call__(self, key, value, query, att_mask, batch_mask, p):
    """
    :param key: DyNet expression of dimensions (input_dim, time) x batch
    :param value: DyNet expression of dimensions (input_dim, time) x batch (None for using value = key)
    :param query: DyNet expression of dimensions (input_dim, time) x batch (None for using query = key)
    :param att_mask: numpy array of dimensions (time, time); pre-transposed
    :param batch_mask: numpy array of dimensions (batch, time)
    :param p: dropout prob
    """
    if value is None or query is None: assert self.is_self_att
    sent_len = key.dim()[0][1]
#    hidden_dim = key.dim()[0][0]
    if self.downsample_factor > 1:
      query_tensor = query.as_tensor() if query else key.as_tensor()
      strided_query = ExpressionSequence(expr_tensor=dy.strided_select(query_tensor, [1,self.downsample_factor], [], []))
      residual = TimeDistributed()(strided_query)
      sent_len_out = len(strided_query)
    else:
      residual = TimeDistributed()(query or key)
      sent_len_out = sent_len
    if self.model_dim!=self.input_dim:
      residual = self.res_shortcut(residual)
      
      
    batch_size = key[0].dim()[1]

    def shape_projection(x):
      total_words = x.dim()[1]
      seq_len = total_words / batch_size
#       temp = dy.reshape(x, (self.model_dim, seq_len), batch_size=batch_size)
#       temp = dy.transpose(temp)
#       temp = dy.reshape(temp, (seq_len, self.dim_per_head), batch_size=batch_size * self.head_count)
      temp = dy.reshape_transpose_reshape(x, (self.model_dim, seq_len), (seq_len, self.dim_per_head), pre_batch_size=batch_size, post_batch_size=batch_size * self.head_count)
      return temp

    # Concatenate all the words together for doing vectorized affine transform
    if self.is_self_att:
      kvq_lin = self.linear_kvq(TimeDistributed()(key))
      key_up = shape_projection(dy.pick_range(kvq_lin, 0, self.head_count * self.dim_per_head)) 
      value_up = shape_projection(dy.pick_range(kvq_lin, self.head_count * self.dim_per_head, 2 * self.head_count * self.dim_per_head))
      query_up = shape_projection(dy.pick_range(kvq_lin, 2 * self.head_count * self.dim_per_head, 3 * self.head_count * self.dim_per_head))
    else:
      key_up = shape_projection(self.linear_keys(TimeDistributed()(key))) 
      value_up = shape_projection(self.linear_values(TimeDistributed()(value)))
      query_up = shape_projection(self.linear_query(TimeDistributed()(query)))

    scaled = query_up * dy.transpose(key_up) # ((T,dim_per_head),head_count*batchsize) * ((dim_per_head,T),head_count*batchsize)
    scaled = scaled / math.sqrt(self.dim_per_head)

    # Apply Mask here
    if not self.ignore_masks:
      if att_mask is not None:
        att_mask_inp = att_mask * -100.0
        if self.broadcast_masks:
          att_mask_inp = np.asarray(np.broadcast_to(att_mask_inp[:,:,np.newaxis], (sent_len, sent_len, self.head_count*batch_size)))
          scaled += dy.inputTensor(att_mask_inp, batched=True)
        else:    
          scaled += dy.inputTensor(att_mask_inp)
      if batch_mask is not None:
        # reshape (batch, time) -> (time, head_count*batch), then *-100
        inp = np.resize(np.broadcast_to(batch_mask.T[:,np.newaxis,:],
                                                           (sent_len, self.head_count, batch_size)), 
                                           (1, sent_len, self.head_count*batch_size)) \
                                           * -100
        if self.broadcast_masks:
          inp = np.asarray(np.broadcast_to(inp, (sent_len, sent_len, self.head_count*batch_size)))
        mask_expr = dy.inputTensor(inp, batched=True)
        scaled += mask_expr

    # Computing Softmax here.
    attn = dy.softmax(scaled, d=1)

    # Applying dropout to attention
    drop_attn = dy.dropout(attn, p)

    # Computing weighted attention score
    attn_prod = drop_attn * value_up
    
    if self.downsample_factor > 1:
      attn_prod = dy.strided_select(attn_prod, [self.downsample_factor], [], [])
    
    if self.plot_attention:
      import matplotlib.pyplot as plt
      assert batch_size==1
      for i in range(attn.dim()[1]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(dy.pick_batch_elem(attn, i).npvalue())
#         im = ax.get_images()
#         extent =  im[0].get_extent()
#         ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.0)
        ax.set_aspect('auto')
        fig.savefig("{}.sent_{}.head_{}.png".format(self.plot_attention, self.plot_attention_counter, i),
                    dpi=1200)
        fig.clf()
        plt.close('all')
      self.plot_attention_counter += 1

    # Reshaping the attn_prod to input query dimensions
#     temp = dy.reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), batch_size=batch_size)
#     temp = dy.transpose(temp)
#     out = dy.reshape(temp, (self.model_dim,), batch_size=batch_size*sent_len_out)
    out = dy.reshape_transpose_reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), (self.model_dim,), pre_batch_size=batch_size, post_batch_size=batch_size*sent_len_out)

    # Adding dropout and layer normalization
    res = dy.dropout(out, p) + residual
    ret = self.layer_norm(res)
    return ret


class TransformerEncoderLayer(object):
  def __init__(self, hidden_dim, model, head_count=8, ff_hidden_dim=2048, downsample_factor=1,
               input_dim=None, diagonal_mask_width=None, mask_self=False, ignore_masks=False, broadcast_masks=False,
               plot_attention=None):
    self.self_attn = MultiHeadedAttention(head_count, hidden_dim, model, downsample_factor, 
                                          input_dim=input_dim, is_self_att=True, ignore_masks=ignore_masks, 
                                          broadcast_masks=broadcast_masks, plot_attention=plot_attention)
    self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_hidden_dim, model)  # Feed Forward
    self.head_count = head_count
    self.downsample_factor = downsample_factor
    self.diagonal_mask_width = diagonal_mask_width
    if diagonal_mask_width: assert diagonal_mask_width%2==1
    self.mask_self = mask_self

  def set_dropout(self, dropout):
    self.dropout = dropout

  def transduce(self, x):
    seq_len = len(x)
    batch_size = x[0].dim()[1]

    att_mask = None
    if self.diagonal_mask_width is not None or self.mask_self:
      if self.diagonal_mask_width is None:
        att_mask = np.zeros((seq_len,seq_len))
      else:
        att_mask = np.ones((seq_len, seq_len))
        for i in range(seq_len):
          from_i = max(0, i-self.diagonal_mask_width//2)
          to_i = min(seq_len, i+self.diagonal_mask_width//2+1)
          att_mask[from_i:to_i,from_i:to_i] = 0.0

      if self.mask_self:
        for i in range(seq_len):
            att_mask[i,i] = 1.0
      
        
    mid = self.self_attn(key=x, value=None, query=None, att_mask=att_mask, batch_mask=x.mask.np_arr if x.mask else None, p=self.dropout)
    if self.downsample_factor > 1:
      seq_len = int(math.ceil(seq_len / float(self.downsample_factor)))
    out = self.feed_forward(mid, p=self.dropout)

    out_mask = x.mask
    if self.downsample_factor > 1 and out_mask is not None:
      out_mask = out_mask.lin_subsampled(reduce_factor = self.downsample_factor)
    
    return ExpressionSequence(expr_tensor=dy.reshape(out, (out.dim()[0][0], seq_len), batch_size=batch_size), mask=out_mask)



class TransformerSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!TransformerSeqTransducer'

  def __init__(self, yaml_context, input_dim=512, layers=1, hidden_dim=512, 
               head_count=8, ff_hidden_dim=2048, dropout=None, 
               downsample_factor=1, diagonal_mask_width=None, mask_self=False,
               ignore_masks=False, broadcast_masks=False, plot_attention=None):
    register_handler(self)
    param_col = yaml_context.dynet_param_collection.param_col
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout or yaml_context.dropout
    self.layers = layers
    self.modules = []
    for layer_i in range(layers):
      if plot_attention is not None:
        plot_attention_layer = "{}.layer_{}".format(plot_attention, layer_i)
      else:
        plot_attention_layer = None
      self.modules.append(TransformerEncoderLayer(hidden_dim, param_col, 
                                                  downsample_factor=downsample_factor, 
                                                  input_dim=input_dim if layer_i==0 else hidden_dim,
                                                  head_count=head_count, ff_hidden_dim=ff_hidden_dim,
                                                  diagonal_mask_width=diagonal_mask_width,
                                                  mask_self=mask_self,
                                                  ignore_masks=ignore_masks,
                                                  broadcast_masks=broadcast_masks,
                                                  plot_attention=plot_attention_layer))

  def __call__(self, sent):
    for module in self.modules:
      enc_sent = module.transduce(sent)
      sent = enc_sent
    self._final_states = [FinalTransducerState(sent[-1])]
    return sent

  @handle_xnmt_event
  def on_set_train(self, val):
    for module in self.modules:
      module.set_dropout(self.dropout if val else 0.0)



