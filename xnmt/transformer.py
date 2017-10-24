import dynet as dy
import math
import numpy as np
from xnmt.expression_sequence import ExpressionSequence
from xnmt.nn import *

MAX_SIZE = 5000
MIN_VAL = -10000   # This value is close to NEG INFINITY


class MultiHeadedAttention(object):
  def __init__(self, head_count, model_dim, model, downsample_factor=1):
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count
    assert downsample_factor >= 1
    self.downsample_factor = downsample_factor

    # Linear Projection of keys
    self.linear_keys = Linear(model_dim, head_count * self.dim_per_head, model)

    # Linear Projection of values
    self.linear_values = Linear(model_dim, head_count * self.dim_per_head, model)

    # Linear Projection of query
    self.linear_query = Linear(model_dim, head_count * self.dim_per_head, model)

    # Layer Norm Module
    self.layer_norm = LayerNorm(model_dim, model)

  def __call__(self, key, value, query, mask, p):
    sent_len = key.dim()[0][1]
    hidden_dim = key.dim()[0][0]
    if self.downsample_factor > 1:
      strided_query = ExpressionSequence(expr_tensor=dy.strided_select(query.as_tensor(), [0, hidden_dim, 1, 0, sent_len, self.downsample_factor]))
      residual = TimeDistributed()(strided_query)
      sent_len_out = len(strided_query)
    else:
      residual = TimeDistributed()(query)
      sent_len_out = sent_len
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
    if mask is not None:
      _, l1, l2, b = mask.shape
      assert(b == batch_size)
      # Following 3 operations are essential to convert a numpy matrix of dimensions mask.shape
      # to the dimensions of scaled tensor in correct way

      # m1 = np.broadcast_to(mask.T, (self.head_count, l, l, batch_size))
      m2 = np.moveaxis(mask, [0, 1, 2], [3, 0, 1])
      m3 = (m2.reshape(l1, l2, -1) * MIN_VAL) + 1  # Convert all 0's to 1's and 0's to MIN_VAL+1
      new_mask = dy.inputTensor(m3, batched=True)
      scaled = dy.cmult(scaled, new_mask)

    # Computing Softmax here. Doing double transpose here, as softmax in dynet is applied to each column
    # May be Optimized ? // Dynet Tricks ??
    attn = dy.transpose(dy.softmax(dy.transpose(scaled)))

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
  def __init__(self, size, rnn_size, model, head_count=8, hidden_size=2048, downsample_factor=1):
    self.self_attn = MultiHeadedAttention(head_count, size, model, downsample_factor)  # Self Attention
    self.feed_forward = PositionwiseFeedForward(size, hidden_size, model)  # Feed Forward
    self.head_count = head_count
    self.downsample_factor = downsample_factor

  def set_dropout(self, dropout):
    self.dropout = dropout

  def transduce(self, input):
    seq_len = len(input)
    batch_size = input[0].dim()[1]

    m_src = None
    if input.mask is not None:
        m_src = np.broadcast_to(input.mask.np_arr.T, (self.head_count, seq_len, seq_len, batch_size))

    mid = self.self_attn(input, input, input, mask=m_src, p=self.dropout)
    if self.downsample_factor > 1:
      seq_len = int(math.ceil(seq_len / float(self.downsample_factor)))
    out = self.feed_forward(mid, p=self.dropout)

#    assert (np.isnan(out.npvalue()).any() == False)  # Check for Nan
    out_list = expr_to_sequence(out, seq_len, batch_size)
    return out_list



