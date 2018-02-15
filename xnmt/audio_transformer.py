import logging
from xnmt.embedder import PositionEmbedder
yaml_logger = logging.getLogger('yaml')
from collections.abc import Sequence
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import dynet as dy

from simple_settings import settings

from xnmt.expression_sequence import ExpressionSequence
from xnmt.mlp import MLP
from xnmt.nn import LayerNorm, Linear, PositionwiseFeedForward, TimeDistributed, PositionwiseLinear, PositionwiseConv
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
from xnmt.events import register_handler, handle_xnmt_event

MAX_SIZE = 5000
MIN_VAL = -10000   # This value is close to NEG INFINITY

class MultiHeadedAttention(object):
  def __init__(self, head_count, model_dim, model, downsample_factor=1, input_dim=None, 
               is_self_att=False, ignore_masks=False, plot_attention=None,
               diag_gauss_mask=False, square_mask_std=False, downsampling_method="skip",
               pos_matrix=False, double_pos_emb=False, glorot_gain=1.0, desc=None):
    """
    :param head_count: number of self-att heads
    :param model_dim: 
    :param model: dynet param collection
    :param downsample_factor:
    :param input_dim:
    :param is_self_att: if True, expect key=query=value
    :param ignore_masks: don't apply any masking
    :param plot_attention: None or path to directory to write plots to
    :param diag_gauss_mask: False to disable, otherwise a float denoting the std of the mask
    :param downsampling_method: how to perform downsampling (reshape|skip)
    """
    if diag_gauss_mask:
      register_handler(self)
    if input_dim is None: input_dim = model_dim
    self.input_dim = input_dim
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count
    assert downsample_factor >= 1
    self.downsample_factor = downsample_factor
    self.downsampling_method = downsampling_method
    self.plot_attention = plot_attention
    self.plot_attention_counter = 0
    self.desc = desc
    
    self.ignore_masks = ignore_masks
    self.diag_gauss_mask = diag_gauss_mask
    self.square_mask_std = square_mask_std
    
    self.is_self_att = is_self_att
    
    if is_self_att:
      self.linear_kvq = Linear(input_dim if self.downsampling_method!="reshape" else input_dim * downsample_factor,
                               head_count * self.dim_per_head * 3, model, glorot_gain=glorot_gain)
      
    else:
      self.linear_keys = Linear(input_dim if self.downsampling_method!="reshape" else input_dim * downsample_factor, head_count * self.dim_per_head, model, glorot_gain=glorot_gain)
      self.linear_values = Linear(input_dim if self.downsampling_method!="reshape" else input_dim * downsample_factor, head_count * self.dim_per_head, model, glorot_gain=glorot_gain)
      self.linear_query = Linear(input_dim if self.downsampling_method!="reshape" else input_dim * downsample_factor, head_count * self.dim_per_head, model, glorot_gain=glorot_gain)
    
    if self.diag_gauss_mask:
      if self.diag_gauss_mask=="rand":
        rand_init = np.exp((np.random.random(size=(self.head_count,)))*math.log(1000))
        self.diag_gauss_mask_sigma = model.add_parameters(dim=(1,1,self.head_count), init=dy.NumpyInitializer(rand_init))
      else:
        self.diag_gauss_mask_sigma = model.add_parameters(dim=(1,1,self.head_count), init=dy.ConstInitializer(self.diag_gauss_mask))

    # Layer Norm Module
    self.layer_norm = LayerNorm(model_dim, model)
    
    if self.downsampling_method=="reshape":
      if model_dim != input_dim * downsample_factor: self.res_shortcut = PositionwiseLinear(input_dim * downsample_factor, model_dim, model, glorot_gain=glorot_gain)
    else:
      if model_dim != input_dim: self.res_shortcut = PositionwiseLinear(input_dim, model_dim, model, glorot_gain=glorot_gain)
    
    self.pos_matrix = pos_matrix
    if pos_matrix=='shallow':
      self.pos_matrix_p = model.add_parameters(dim=(1,1,70,self.head_count), init=dy.GlorotInitializer(gain=glorot_gain))
    elif pos_matrix:
      self.pos_matrix_p = model.add_parameters(dim=(1,1,70,self.dim_per_head*self.head_count), init=dy.GlorotInitializer(gain=glorot_gain))
 
    self.double_pos_emb = double_pos_emb
    if double_pos_emb:
      self.double_pos_emb_p1 = model.add_parameters(dim=(double_pos_emb, self.dim_per_head, self.head_count), init=dy.NormalInitializer(mean=1.0, var=0.001))
      self.double_pos_emb_p2 = model.add_parameters(dim=(double_pos_emb, self.dim_per_head, self.head_count), init=dy.NormalInitializer(mean=1.0, var=0.001))
 
  def plot_att_mat(self, mat, filename, dpi=1200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(mat)
    ax.set_aspect('auto')
    fig.savefig(filename, dpi=dpi)
    fig.clf()
    plt.close('all')

  def shape_projection(self, x, batch_size):
    total_words = x.dim()[1]
    seq_len = total_words / batch_size
    out = dy.reshape(x, (self.model_dim, seq_len), batch_size=batch_size)
    out = dy.transpose(out)
    return dy.reshape(out, (seq_len, self.dim_per_head), batch_size=batch_size * self.head_count)
#     return dy.reshape_transpose_reshape(x, (self.model_dim, seq_len), (seq_len, self.dim_per_head), pre_batch_size=batch_size, post_batch_size=batch_size * self.head_count)

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
    batch_size = key[0].dim()[1]

    if self.downsample_factor > 1:
      if self.downsampling_method == "skip":
        query_tensor = query.as_tensor() if query else key.as_tensor()
        strided_query = ExpressionSequence(expr_tensor=dy.strided_select(query_tensor, [1,self.downsample_factor], [], []))
        residual = TimeDistributed()(strided_query)
        sent_len_out = len(strided_query)
      else:
        assert self.downsampling_method == "reshape"
        if sent_len % self.downsample_factor != 0:
          raise ValueError("For 'reshape' downsampling, sequence lengths must be multiples of the downsampling factor. Configure batcher accordingly.")
        if batch_mask is not None: batch_mask = batch_mask[:,::self.downsample_factor]
        sent_len_out = sent_len // self.downsample_factor
        sent_len = sent_len_out
        out_mask = key.mask
        if self.downsample_factor > 1 and out_mask is not None:
          out_mask = out_mask.lin_subsampled(reduce_factor = self.downsample_factor)

        if query:
          query = ExpressionSequence(expr_tensor=dy.reshape(query.as_tensor(), (query.dim()[0][0] * self.downsample_factor, query.dim()[0][1] / self.downsample_factor), batch_size = batch_size),
                                     mask=out_mask)
        if key:
          key = ExpressionSequence(expr_tensor=dy.reshape(key.as_tensor(), (key.dim()[0][0] * self.downsample_factor, key.dim()[0][1] / self.downsample_factor), batch_size = batch_size),
                                   mask=out_mask)
        if value:
          value = ExpressionSequence(expr_tensor=dy.reshape(value.as_tensor(), (value.dim()[0][0] * self.downsample_factor, value.dim()[0][1] / self.downsample_factor), batch_size = batch_size),
                                     mask=out_mask)
        residual = TimeDistributed()(query or key)
    else:
      residual = TimeDistributed()(query or key)
      sent_len_out = sent_len
    if self.downsampling_method=="reshape":
      if self.model_dim!=self.input_dim*self.downsample_factor:
        residual = self.res_shortcut(residual)
    else:
      if self.model_dim!=self.input_dim:
        residual = self.res_shortcut(residual)
      
    # Concatenate all the words together for doing vectorized affine transform
    if self.is_self_att:
      kvq_lin = self.linear_kvq(TimeDistributed()(key))
      key_up = self.shape_projection(dy.pick_range(kvq_lin, 0, self.head_count * self.dim_per_head), batch_size) 
      value_up = self.shape_projection(dy.pick_range(kvq_lin, self.head_count * self.dim_per_head, 2 * self.head_count * self.dim_per_head), batch_size)
      query_up = self.shape_projection(dy.pick_range(kvq_lin, 2 * self.head_count * self.dim_per_head, 3 * self.head_count * self.dim_per_head), batch_size)
    else:
      key_up = self.shape_projection(self.linear_keys(TimeDistributed()(key)), batch_size) 
      value_up = self.shape_projection(self.linear_values(TimeDistributed()(value)), batch_size)
      query_up = self.shape_projection(self.linear_query(TimeDistributed()(query)), batch_size)
      
    if self.double_pos_emb:
      emb1 = dy.pick_range(dy.parameter(self.double_pos_emb_p1), 0,sent_len)
      emb2 = dy.pick_range(dy.parameter(self.double_pos_emb_p2), 0,sent_len)
      key_up = dy.reshape(key_up, (sent_len, self.dim_per_head, self.head_count), batch_size=batch_size)
      key_up = dy.concatenate_cols([dy.cmult(key_up, emb1), dy.cmult(key_up, emb2)])
      key_up = dy.reshape(key_up, (sent_len, self.dim_per_head*2), batch_size=self.head_count*batch_size)
      query_up = dy.reshape(query_up, (sent_len, self.dim_per_head, self.head_count), batch_size=batch_size)
      query_up = dy.concatenate_cols([dy.cmult(query_up, emb2), dy.cmult(query_up, -emb1)])
      query_up = dy.reshape(query_up, (sent_len, self.dim_per_head*2), batch_size=self.head_count*batch_size)

#     scaled = query_up * dy.transpose(key_up) / math.sqrt(self.dim_per_head)
    if self.pos_matrix=='shallow':
      def map_fnc(v):
        if v<1: return 0
        elif v<2: return 1
        elif v<4: return 2
        elif v<8: return 3
        elif v<16: return 4
        elif v<32: return 5
        elif v<64: return 6
        elif v<128: return 7
        elif v<256: return 8
        else: return 9
      #map_fnc = lambda v: min(10,int(math.log2(1+v)))
      # TODO: this is apparently very slow and needs to be re-designed
      #indices_0 = [i for i in range(sent_len) for j in range(sent_len)] * 7
      #indices_1 = [i for i in range(sent_len) for j in range(sent_len)] * 7
      #indices_2 = [   map_fnc(math.fabs(i-j)) for i in range(sent_len) for j in range(sent_len)] +\
      #            [10+map_fnc(max(i-j, 0))    for i in range(sent_len) for j in range(sent_len)] +\
      #            [20+map_fnc(max(j-i, 0))    for i in range(sent_len) for j in range(sent_len)] +\
      #            [30+map_fnc(i)              for i in range(sent_len) for j in range(sent_len)] +\
      #            [40+map_fnc(sent_len-i)     for i in range(sent_len) for j in range(sent_len)] +\
      #            [50+map_fnc(j)              for i in range(sent_len) for j in range(sent_len)] +\
      #            [60+map_fnc(sent_len-j)     for i in range(sent_len) for j in range(sent_len)]
      #values = [1.0] * (sent_len * sent_len * 7)
      one_hot_pos_matrix = dy.ones((sent_len, sent_len, 70))
      #one_hot_pos_matrix = dy.sparse_inputTensor([indices_0, indices_1, indices_2],
      #                                           values,
      #                                           shape=(sent_len, sent_len, 70))
      embedded_pos_matrix = dy.conv2d(one_hot_pos_matrix,dy.parameter(self.pos_matrix_p),stride=(1,1))
      scaled = query_up * dy.transpose(key_up / math.sqrt(self.dim_per_head))
      scaled = dy.reshape(scaled, (sent_len, sent_len, self.head_count), batch_size=batch_size)
      scaled = scaled + embedded_pos_matrix
      scaled = dy.reshape(scaled, (sent_len, sent_len), batch_size=self.head_count*batch_size)
    elif self.pos_matrix:
      # this needs a crazy amount of memory and should probably be avoided
      left = dy.reshape(query_up, (sent_len,1), batch_size=self.dim_per_head*self.head_count*batch_size)
      right = dy.reshape(query_up, (1,sent_len), batch_size=self.dim_per_head*self.head_count*batch_size)
      m = dy.reshape(left * right, (sent_len * sent_len, self.dim_per_head, self.head_count), batch_size=batch_size)
      #  0.. 9: |pos_i - pos_j|
      # 10..19: max(pos_i - pos_j, 0)
      # 20..29: max(pos_j - pos_i, 0)
      # 30..39: pos_i
      # 40..49: len-pos_i
      # 50..59: pos_j
      # 60..69: len-pos_j
      map_fnc = lambda v: min(10,int(math.log2(1+v)))
      indices_0 = [i for i in range(sent_len) for j in range(sent_len)] * 7
      indices_1 = [i for i in range(sent_len) for j in range(sent_len)] * 7
      indices_2 = [   map_fnc(math.fabs(i-j)) for i in range(sent_len) for j in range(sent_len)] +\
                  [10+map_fnc(max(i-j, 0))    for i in range(sent_len) for j in range(sent_len)] +\
                  [20+map_fnc(max(j-i, 0))    for i in range(sent_len) for j in range(sent_len)] +\
                  [30+map_fnc(i)              for i in range(sent_len) for j in range(sent_len)] +\
                  [40+map_fnc(sent_len-i)     for i in range(sent_len) for j in range(sent_len)] +\
                  [50+map_fnc(j)              for i in range(sent_len) for j in range(sent_len)] +\
                  [60+map_fnc(sent_len-j)     for i in range(sent_len) for j in range(sent_len)]
      values = [1.0] * (sent_len * sent_len * 7)
      one_hot_pos_matrix = dy.sparse_inputTensor([indices_0, indices_1, indices_2],
                                                 values,
                                                 shape=(sent_len, sent_len, 70))
      embedded_pos_matrix = dy.conv2d(one_hot_pos_matrix,dy.parameter(self.pos_matrix_p),stride=(1,1))
      embedded_pos_matrix = dy.reshape(embedded_pos_matrix, (sent_len*sent_len,self.dim_per_head,self.head_count),)
      m2 = dy.cmult(m, embedded_pos_matrix)
      scaled = dy.reshape(dy.sum_dim(m2, d=[1]), (sent_len, sent_len), batch_size=self.head_count*batch_size)
    else:
      scaled = query_up * dy.transpose(key_up / math.sqrt(self.dim_per_head)) # scale before the matrix multiplication to save memory

    # Apply Mask here
    if not self.ignore_masks:
      if att_mask is not None:
        att_mask_inp = att_mask * -100.0
        if self.downsampling_method=="reshape" and self.downsample_factor>1:
          att_mask_inp = att_mask_inp[::self.downsample_factor,::self.downsample_factor]
        scaled += dy.inputTensor(att_mask_inp)
      if batch_mask is not None:
        # reshape (batch, time) -> (time, head_count*batch), then *-100
        inp = np.resize(np.broadcast_to(batch_mask.T[:,np.newaxis,:],
                                        (sent_len, self.head_count, batch_size)), 
                        (1, sent_len, self.head_count*batch_size)) \
              * -100
        mask_expr = dy.inputTensor(inp, batched=True)
        scaled += mask_expr
      if self.diag_gauss_mask:
        diag_growing = np.zeros((sent_len, sent_len, self.head_count))
        for i in range(sent_len):
          for j in range(sent_len):
            diag_growing[i,j,:] = -(i-j)**2 / 2.0
        e_diag_gauss_mask = dy.inputTensor(diag_growing)
        e_sigma = dy.parameter(self.diag_gauss_mask_sigma)
        if self.square_mask_std:
          e_sigma = dy.square(e_sigma)
        e_sigma_sq_inv = dy.cdiv(dy.ones(e_sigma.dim()[0], batch_size=batch_size), dy.square(e_sigma))
        e_diag_gauss_mask_final = dy.cmult(e_diag_gauss_mask, e_sigma_sq_inv)
        scaled += dy.reshape(e_diag_gauss_mask_final, (sent_len, sent_len), batch_size=batch_size * self.head_count)
    
    # Computing Softmax here.
    attn = dy.softmax(scaled, d=1)
    if settings.LOG_ATTENTION:
      yaml_logger.info({"key":"selfatt_mat_ax0", "value":np.sum(attn.value(),axis=0).dumps(), "desc":self.desc})
      yaml_logger.info({"key":"selfatt_mat_ax1", "value":np.sum(attn.value(),axis=1).dumps(), "desc":self.desc})

    # Applying dropout to attention
    if p>0.0:
      drop_attn = dy.dropout(attn, p)
    else:
      drop_attn = attn

    # Computing weighted attention score
    attn_prod = drop_attn * value_up
    
    if self.downsample_factor > 1 and self.downsampling_method == "skip":
      attn_prod = dy.strided_select(attn_prod, [self.downsample_factor], [], [])
    

    # Reshaping the attn_prod to input query dimensions
    out = dy.reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), batch_size=batch_size)
    out = dy.transpose(out)
    out = dy.reshape(out, (self.model_dim,), batch_size=batch_size*sent_len_out)
#     out = dy.reshape_transpose_reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), (self.model_dim,), pre_batch_size=batch_size, post_batch_size=batch_size*sent_len_out)

    if self.plot_attention:
      assert batch_size==1
      mats = []
      for i in range(attn.dim()[1]):
        mats.append(dy.pick_batch_elem(attn, i).npvalue())
        self.plot_att_mat(mats[-1], 
                          "{}.sent_{}.head_{}.png".format(self.plot_attention, self.plot_attention_counter, i),
                          300)
      avg_mat = np.average(mats,axis=0)
      self.plot_att_mat(avg_mat, 
                        "{}.sent_{}.head_avg.png".format(self.plot_attention, self.plot_attention_counter),
                        300)
      in_val = value or key
      cosim_before = cosine_similarity(in_val.as_tensor().npvalue().T)
      self.plot_att_mat(cosim_before, 
                        "{}.sent_{}.cosim_before.png".format(self.plot_attention, self.plot_attention_counter),
                        600)
      cosim_after = cosine_similarity(out.npvalue().T)
      self.plot_att_mat(cosim_after, 
                        "{}.sent_{}.cosim_after.png".format(self.plot_attention, self.plot_attention_counter),
                        600)
      self.plot_attention_counter += 1
      
    # Adding dropout and layer normalization
    if p>0.0:
      res = dy.dropout(out, p) + residual
    else:
      res = out + residual
    ret = self.layer_norm(res)
    return ret

  @handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    yaml_logger.info({"key":"self_att_mask_var: ", "val":[float(x) for x in list(self.diag_gauss_mask_sigma.as_array().flat)], "desc":self.desc})

class TransformerEncoderLayer(object):
  def __init__(self, hidden_dim, model, head_count=8, ff_hidden_dim=2048, downsample_factor=1,
               input_dim=None, diagonal_mask_width=None, mask_self=False, ignore_masks=False,
               plot_attention=None, nonlinearity="rectify", diag_gauss_mask=False,
               square_mask_std=False, downsampling_method="skip", pos_matrix=False,
               double_pos_emb=False, ff_window=1, glorot_gain=1.0, desc=None):
    self.self_attn = MultiHeadedAttention(head_count, hidden_dim, model, downsample_factor, 
                                          input_dim=input_dim, is_self_att=True, ignore_masks=ignore_masks, 
                                          plot_attention=plot_attention,
                                          diag_gauss_mask=diag_gauss_mask, square_mask_std=square_mask_std,
                                          downsampling_method=downsampling_method,
                                          pos_matrix=pos_matrix,
                                          glorot_gain=glorot_gain,
                                          double_pos_emb=double_pos_emb,
                                          desc=desc)
    self.ff_window = ff_window
    if ff_window==1:
      self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_hidden_dim, model, nonlinearity=nonlinearity,
                                                glorot_gain=glorot_gain)
    else:
      self.feed_forward = PositionwiseConv(hidden_dim, ff_hidden_dim, ff_window, model, nonlinearity=nonlinearity,
                                                glorot_gain=glorot_gain)
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
    if self.ff_window > 1:
      hidden_dim = mid.dim()[0][0]
      mid = dy.reshape(mid, (hidden_dim, seq_len), batch_size=batch_size)
      out = self.feed_forward(mid, p=self.dropout)
      out = dy.reshape(out, (hidden_dim,), batch_size=seq_len * batch_size)
    else:
      out = self.feed_forward(mid, p=self.dropout)

    out_mask = x.mask
    if self.downsample_factor > 1 and out_mask is not None:
      out_mask = out_mask.lin_subsampled(reduce_factor = self.downsample_factor)
    
    self._recent_output = out
    return ExpressionSequence(expr_tensor=dy.reshape(out, (out.dim()[0][0], seq_len), batch_size=batch_size), mask=out_mask)



class TransformerSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!TransformerSeqTransducer'

  def __init__(self, exp_global=Ref(Path("exp_global")), input_dim=512, layers=1, hidden_dim=512, 
               head_count=8, ff_hidden_dim=2048, dropout=None, 
               downsample_factor=1, diagonal_mask_width=None, mask_self=False,
               ignore_masks=False, plot_attention=None, nonlinearity="rectify",
               pos_encoding_type=None, pos_encoding_combine="concat",
               pos_encoding_size=40, max_len=1500,
               pos_matrix=False, diag_gauss_mask=False, square_mask_std=False,
               downsampling_method="skip", double_pos_emb=False, ff_window=1,
               glorot_gain=None):
    """
    :param pos_encoding_type: None, trigonometric, embedding, mlp
    :param pos_encoding_combine: add, concat
    """
    register_handler(self)
    param_col = exp_global.dynet_param_collection.param_col
    glorot_gain = glorot_gain or exp_global.glorot_gain
    self.input_dim = input_dim = (input_dim + (pos_encoding_size if (pos_encoding_type and pos_encoding_combine=="concat") else 0))
    self.hidden_dim = hidden_dim
    self.dropout = dropout or exp_global.dropout
    self.layers = layers
    self.modules = []
    self.pos_encoding_type = pos_encoding_type
    self.pos_encoding_combine = pos_encoding_combine
    self.pos_encoding_size = pos_encoding_size
    self.max_len = max_len
    self.position_encoding_block = None
    if self.pos_encoding_type=="embedding":
      self.positional_embedder = PositionEmbedder(max_pos=self.max_len,
                                                  exp_global=exp_global,
                                                  emb_dim=input_dim if self.pos_encoding_combine=="add" else self.pos_encoding_size)
    elif self.pos_encoding_type=="mlp":
      self.positional_mlp = MLP(input_dim=3,
                                hidden_dim=input_dim if self.pos_encoding_combine=="add" else self.pos_encoding_size,
                                output_dim=input_dim if self.pos_encoding_combine=="add" else self.pos_encoding_size,
                                model=param_col,
                                layers=1)
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
                                                  plot_attention=plot_attention_layer,
                                                  nonlinearity=nonlinearity,
                                                  diag_gauss_mask=diag_gauss_mask,
                                                  square_mask_std=square_mask_std,
                                                  downsampling_method=downsampling_method,
                                                  pos_matrix=pos_matrix,
                                                  double_pos_emb=double_pos_emb,
                                                  ff_window=ff_window,
                                                  glorot_gain=glorot_gain[layer_i] if isinstance(glorot_gain,Sequence) else glorot_gain,
                                                  desc=f"layer_{layer_i}"))

  def __call__(self, sent):
    if self.pos_encoding_type == "trigonometric":
      if self.position_encoding_block is None or self.position_encoding_block.shape[2] < len(sent):
        self.initialize_position_encoding(int(len(sent) * 1.2), self.input_dim if self.pos_encoding_combine=="add" else self.pos_encoding_size)
      encoding = dy.inputTensor(self.position_encoding_block[0, :, :len(sent)])
    elif self.pos_encoding_type == "embedding":
      encoding = self.positional_embedder.embed_sent(len(sent)).as_tensor()
    elif self.pos_encoding_type == "mlp":
      inp = dy.inputTensor(np.asarray([[i/1000.0  for i in range(len(sent))],[(len(sent)-i)/1000.0 for i in range(len(sent))], [(len(sent))/1000.0 for _ in range(len(sent))]] ), batched=True)
      mlp_out = self.positional_mlp(inp)
      encoding = dy.reshape(mlp_out, (self.input_dim if self.pos_encoding_combine=="add" else self.pos_encoding_size, len(sent)))
    if self.pos_encoding_type:
      if self.pos_encoding_combine=="add":
        sent = ExpressionSequence(expr_tensor=sent.as_tensor() + encoding, mask=sent.mask)
      else: # concat
        sent = ExpressionSequence(expr_tensor=dy.concatenate([sent.as_tensor(), encoding]),
                                  mask=sent.mask)
      
    elif self.pos_encoding_type:
      raise ValueError(f"unknown encoding type {self.pos_encoding_type}")
    for module in self.modules:
      enc_sent = module.transduce(sent)
      self.last_output.append(module._recent_output)
      sent = enc_sent
    self._final_states = [FinalTransducerState(sent[-1])]
    return sent

  def get_final_states(self):
    return self._final_states

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.last_output = []

  @handle_xnmt_event
  def on_set_train(self, val):
    for module in self.modules:
      module.set_dropout(self.dropout if val else 0.0)

  @handle_xnmt_event
  def on_collect_recent_outputs(self):
    return [(self, o) for o in self.last_output]

  def initialize_position_encoding(self, length, n_units):
    # Implementation in the Google tensor2tensor repo
    channels = n_units
    position = np.arange(length, dtype='f')
    num_timescales = channels // 2
    log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
    inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.reshape(signal, [1, length, channels])
    self.position_encoding_block = np.transpose(signal, (0, 2, 1))

