import dynet as dy
import math
import numpy as np
from xnmt.expression_sequence import ExpressionSequence
from xnmt.nn import LayerNorm, Linear, PositionwiseFeedForward, TimeDistributed, PositionwiseLinear

MAX_SIZE = 5000
MIN_VAL = -10000   # This value is close to NEG INFINITY


class MultiHeadedAttention(object):
  def __init__(self, head_count, model_dim, model, downsample_factor=1, input_dim=None):
    if input_dim is None: input_dim = model_dim
    self.input_dim = input_dim
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count
    assert downsample_factor >= 1
    self.downsample_factor = downsample_factor

    # Linear Projection of keys
    self.linear_keys = Linear(input_dim, head_count * self.dim_per_head, model)

    # Linear Projection of values
    self.linear_values = Linear(input_dim, head_count * self.dim_per_head, model)

    # Linear Projection of query
    self.linear_query = Linear(input_dim, head_count * self.dim_per_head, model)

    # Layer Norm Module
    self.layer_norm = LayerNorm(model_dim, model)
    
    if model_dim!=input_dim: self.res_shortcut = PositionwiseLinear(input_dim, model_dim, model)

  def __call__(self, key, value, query, att_mask, batch_mask, p):
    """
    :param key: DyNet expression of dimensions (input_dim, time) x batch
    :param value: DyNet expression of dimensions (input_dim, time) x batch
    :param query: DyNet expression of dimensions (input_dim, time) x batch
    :param att_mask: numpy array of dimensions (time, time); pre-transposed
    :param batch_mask: numpy array of dimensions (batch, time)
    :param p: dropout prob
    """
    sent_len = key.dim()[0][1]
    hidden_dim = key.dim()[0][0]
    if self.downsample_factor > 1:
      strided_query = ExpressionSequence(expr_tensor=dy.strided_select(query.as_tensor(), [0, hidden_dim, 1, 0, sent_len, self.downsample_factor]))
      residual = TimeDistributed()(strided_query)
      sent_len_out = len(strided_query)
    else:
      residual = TimeDistributed()(query)
      sent_len_out = sent_len
    if self.model_dim!=self.input_dim:
      residual = self.res_shortcut(residual)
      
      
    batch_size = key[0].dim()[1]

    def shape_projection(x):
      total_words = x.dim()[1]
      seq_len = total_words / batch_size
      temp = dy.reshape(x, (self.model_dim, seq_len), batch_size=batch_size)
      temp = dy.transpose(temp)
      return dy.reshape(temp, (seq_len, self.dim_per_head), batch_size=batch_size * self.head_count)

    # Concatenate all the words together for doing vectorized affine transform
    key_up = shape_projection(self.linear_keys(TimeDistributed()(key)))
    value_up = shape_projection(self.linear_values(TimeDistributed()(value)))
    query_up = shape_projection(self.linear_query(TimeDistributed()(query)))

    scaled = query_up * dy.transpose(key_up)
    scaled = scaled / math.sqrt(self.dim_per_head)

    # Apply Mask here
    if att_mask is not None:
      scaled += dy.inputTensor(att_mask * -100.0)
    if batch_mask is not None:
      # reshape (batch, time) -> (time, head_count*batch), then *-100
      mask_expr = dy.inputTensor(np.resize(np.broadcast_to(batch_mask.T[:,np.newaxis,:],
                                                         (sent_len, self.head_count, batch_size)), 
                                         (1, sent_len, self.head_count*batch_size)) \
                                         * -100,
                                 batched=True)
      scaled += mask_expr

    # Computing Softmax here.
    attn = dy.softmax_rows(scaled)

    # Applying dropout to attention
    drop_attn = dy.dropout(attn, p)

    # Computing weighted attention score
    attn_prod = drop_attn * value_up
    
    if self.downsample_factor > 1:
      attn_prod = dy.strided_select(attn_prod, [0,sent_len,self.downsample_factor])

    # Reshaping the attn_prod to input query dimensions
    temp = dy.reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), batch_size=batch_size)
    temp = dy.transpose(temp)
    out = dy.reshape(temp, (self.model_dim,), batch_size=batch_size*sent_len_out)

    # Adding dropout and layer normalization
    res = dy.dropout(out, p) + residual
    ret = self.layer_norm(res)
    return ret


def expr_to_sequence(expr_, seq_len, batch_size):
  out_list = []
  for i in range(seq_len):
    indexes = map(lambda x: x + i, range(0, seq_len * batch_size, seq_len))
    out_list.append(dy.pick_batch_elems(expr_, indexes))
  return out_list


class TransformerEncoderLayer(object):
  def __init__(self, hidden_dim, model, head_count=8, ff_hidden_dim=2048, downsample_factor=1,
               input_dim=None, diagonal_mask_width=None, mask_self=False):
    self.self_attn = MultiHeadedAttention(head_count, hidden_dim, model, downsample_factor, 
                                          input_dim=input_dim)
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
      
        
    mid = self.self_attn(key=x, value=x, query=x, att_mask=att_mask, batch_mask=x.mask.np_arr if x.mask else None, p=self.dropout)
    if self.downsample_factor > 1:
      seq_len = int(math.ceil(seq_len / float(self.downsample_factor)))
    out = self.feed_forward(mid, p=self.dropout)

#    assert (np.isnan(out.npvalue()).any() == False)  # Check for Nan
    out_list = expr_to_sequence(out, seq_len, batch_size)
    
    out_mask = x.mask
    if self.downsample_factor > 1 and out_mask is not None:
      out_mask = out_mask.lin_subsampled(reduce_factor = self.downsample_factor)
    
    return ExpressionSequence(out_list, mask=out_mask)



