import io

import dynet as dy

from xnmt.transducer import Transducer, FinalTransducerState
from xnmt.input import Input, BaseTextReader
from xnmt.serializer import Serializable
from xnmt.vocab import Vocab
from xnmt.embedder import SimpleWordEmbedder
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.batcher import is_batched

class LatticeNode(object):
  def __init__(self, nodes_prev, nodes_next, value):
    """
    :param nodes_prev: list of integers indicating indices of direct predecessors
    :param nodes_next: list of integers indicating indices of direct successors
    :param value:
    """
    self.nodes_prev = nodes_prev
    self.nodes_next = nodes_next
    self.value = value
  def new_node_with_val(self, value):
    return LatticeNode(self.nodes_prev, self.nodes_next, value)

class Lattice(Input):
  def __init__(self, nodes, add_bwd_connections=False):
    """
    :param nodes: list of LatticeNode objects
    """
    self.nodes = nodes
    if add_bwd_connections:
      self._add_bwd_connections()
    assert len(nodes[0].nodes_prev) == 0
    assert len(nodes[-1].nodes_next) == 0
    for t in range(1,len(nodes)-1):
      assert len(nodes[t].nodes_prev) > 0
      assert len(nodes[t].nodes_next) > 0
    self.mask = None
    self.expr_tensor = None
  def __len__(self):
    return len(self.nodes)
  def __getitem__(self, key):
    return self.nodes[key]
  def get_padded_sent(self, token, pad_len):
    assert pad_len == 0
    return self
  def as_list(self):
    return [node.value for node in self.nodes]
  def as_tensor(self):
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate_cols(self.as_list())
    return self.expr_tensor
  def _add_bwd_connections(self):
    for pos in range(len(self.nodes)):
      for pred_i in self.nodes[pos].nodes_prev:
        self.nodes[pred_i].nodes_next.append(pos)

  def reversed(self):
    rev_nodes = []
    seq_len = len(self.nodes)
    for node in reversed(self.nodes):
      new_node = LatticeNode(nodes_prev = [seq_len - n - 1 for n in node.nodes_next],
                             nodes_next = [seq_len - p - 1 for p in node.nodes_prev],
                             value = node.value)
      rev_nodes.append(new_node)
    return Lattice(rev_nodes)
  
class LatticeTextReader(BaseTextReader, Serializable):
  yaml_tag = u'!LatticeTextReader'
  def __init__(self, vocab=None, use_words=True, use_chars=False, use_pronun_from=None):
    self.vocab = vocab
    self.use_chars = use_chars
    self.use_words = use_words
    self.use_pronun = False
    if use_pronun_from:
      self.use_pronun = {}
      for l in io.open(use_pronun_from):
        spl = l.strip().split()
        word = spl[0]
        pronun = spl[1:]
        assert word not in self.use_pronun
        self.use_pronun[word] = pronun
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    sents = []
    for l in self.iterate_filtered(filename, filter_ids):
      words = l.strip().split()
      if words[0] != Vocab.SS_STR: words.insert(0, Vocab.SS_STR)
      if words[-1] != Vocab.ES_STR: words.append(Vocab.ES_STR)
      mapped_words = [LatticeNode([], [1], self.vocab.convert(words[0]))]
      prev_indices = [0]
      for word in words[1:-1]:
        representations = self.get_representations(word)
        new_prev_indices = []
        for rep in representations:
          for rep_pos in range(len(rep)):
            if rep_pos==0:
              preds = prev_indices
            else:
              preds = [len(mapped_words)-1]
#             print("node", len(mapped_words), rep[rep_pos])
            mapped_words.append(LatticeNode(preds, [], self.vocab.convert(rep[rep_pos])))
          new_prev_indices.append(len(mapped_words)-1)
        prev_indices = new_prev_indices
      mapped_words.append(LatticeNode(prev_indices, [], self.vocab.convert(words[-1])))
      lattice = Lattice(mapped_words, add_bwd_connections=True)
      sents.append(lattice)
    return sents

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param(u"vocab", self.vocab)

  def vocab_size(self):
    return len(self.vocab)
  
  def get_representations(self, word):
    reps = []
    if self.use_words:
      reps.append([word])
    if self.use_chars:
      reps.append(["c_"+char for char in word] + ["c_"])
    if self.use_pronun:
      if word in self.use_pronun:
        reps.append(self.use_pronun[word] + ['__'])
      else:
        print("WARNING: no pronunciation for", word)
    return reps

class LatticeEmbedder(SimpleWordEmbedder, Serializable):
  """
  Simple word embeddings via lookup.
  """

  yaml_tag = u'!LatticeEmbedder'

  def __init__(self, yaml_context, vocab_size, emb_dim = None, word_dropout = 0.0):
    """
    :param vocab_size:
    :param emb_dim:
    :param word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    """
    register_handler(self)
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim or yaml_context.default_layer_dim
    self.word_dropout = word_dropout
    self.embeddings = yaml_context.dynet_param_collection.param_col.add_lookup_parameters((self.vocab_size, self.emb_dim))
    self.word_id_mask = None
    self.train = False
    self.weight_noise = 0.0
    self.fix_norm = None

  def embed_sent(self, sent):
    if is_batched(sent):
      assert len(sent)==1, "LatticeEmbedder requires batch size of 1"
      assert sent.mask is None
      sent = sent[0]
    embedded_nodes = [word.new_node_with_val(self.embed(word.value)) for word in sent]
    return Lattice(nodes=embedded_nodes)



class LatticeLSTMTransducer(Transducer):
  def __init__(self, yaml_context, input_dim, hidden_dim, dropout = 0.0):
    register_handler(self)
    if dropout and dropout > 0.0: raise NotImplementedError()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = yaml_context.dynet_param_collection.param_col

    # [i; f; o; g]
    self.p_Wx_iog = model.add_parameters(dim=(hidden_dim*3, input_dim))
    self.p_Wh_iog = model.add_parameters(dim=(hidden_dim*3, hidden_dim))
    self.p_b_iog  = model.add_parameters(dim=(hidden_dim*3,), init=dy.ConstInitializer(0.0))
    self.p_Wx_f = model.add_parameters(dim=(hidden_dim, input_dim))
    self.p_Wh_f = model.add_parameters(dim=(hidden_dim, hidden_dim))
    self.p_b_f  = model.add_parameters(dim=(hidden_dim,), init=dy.ConstInitializer(1.0))

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, lattice):
    Wx_iog = dy.parameter(self.p_Wx_iog)
    Wh_iog = dy.parameter(self.p_Wh_iog)
    b_iog = dy.parameter(self.p_b_iog)
    Wx_f = dy.parameter(self.p_Wx_f)
    Wh_f = dy.parameter(self.p_Wh_f)
    b_f = dy.parameter(self.p_b_f)
    h = []
    c = []
    for x_t in lattice:
      i_ft_list = []
      if len(x_t.nodes_prev)==0:
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, x_t.value])
      else:
        h_tilde = sum(h[pred] for pred in x_t.nodes_prev)
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, x_t.value, Wh_iog, h_tilde])
        for pred in x_t.nodes_prev:
          i_ft_list.append(dy.logistic(dy.affine_transform([b_f, Wx_f, x_t.value, Wh_f, h[pred]])))
      i_ait = dy.pick_range(tmp_iog, 0, self.hidden_dim)
#       i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pick_range(tmp_iog, self.hidden_dim, self.hidden_dim*2)
      i_agt = dy.pick_range(tmp_iog, self.hidden_dim*2, self.hidden_dim*3)

      i_it = dy.logistic(i_ait)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if len(x_t.nodes_prev)==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        fc = dy.cmult(i_ft_list[0], c[x_t.nodes_prev[0]])
        for i in range(1,len(x_t.nodes_prev)):
          fc += dy.cmult(i_ft_list[i], c[x_t.nodes_prev[i]])
        c.append(fc + dy.cmult(i_it, i_gt))
      h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
#     self._final_states = [FinalTransducerState(dy.reshape(h[-1], (self.hidden_dim,)),\
#                                                dy.reshape(c[-1], (self.hidden_dim,)))]
    self._final_states = [FinalTransducerState(h[-1], c[-1])]
    return Lattice(nodes=[node_t.new_node_with_val(h_t) for node_t, h_t in zip(lattice.nodes,h)])


#   def seq_lstm_call(self, xs):
#     Wx = dy.parameter(self.p_Wx)
#     Wh = dy.parameter(self.p_Wh)
#     b = dy.parameter(self.p_b)
#     h = []
#     c = []
#     for i, x_t in enumerate(xs):
#       if i==0:
#         tmp = dy.affine_transform([b, Wx, x_t])
#       else:
#         tmp = dy.affine_transform([b, Wx, x_t, Wh, h[-1]])
#       i_ait = dy.pick_range(tmp, 0, self.hidden_dim)
#       i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
#       i_aot = dy.pick_range(tmp, self.hidden_dim*2, self.hidden_dim*3)
#       i_agt = dy.pick_range(tmp, self.hidden_dim*3, self.hidden_dim*4)
#       i_it = dy.logistic(i_ait)
#       i_ft = dy.logistic(i_aft + 1.0)
#       i_ot = dy.logistic(i_aot)
#       i_gt = dy.tanh(i_agt)
#       if i==0:
#         c.append(dy.cmult(i_it, i_gt))
#       else:
#         c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
#       h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
#     return h

class BiLatticeLSTMTransducer(Transducer, Serializable):
  yaml_tag = u'!BiLatticeLSTMTransducer'
  
  def __init__(self, yaml_context, layers=1, input_dim=None, hidden_dim=None, dropout=None):
    register_handler(self)
    self.num_layers = layers
    input_dim = input_dim or yaml_context.default_layer_dim
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or yaml_context.dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = [LatticeLSTMTransducer(yaml_context, input_dim, hidden_dim/2, dropout)]
    self.backward_layers = [LatticeLSTMTransducer(yaml_context, input_dim, hidden_dim/2, dropout)]
    self.forward_layers += [LatticeLSTMTransducer(yaml_context, hidden_dim, hidden_dim/2, dropout) for _ in range(layers-1)]
    self.backward_layers += [LatticeLSTMTransducer(yaml_context, hidden_dim, hidden_dim/2, dropout) for _ in range(layers-1)]

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, lattice):
    # first layer
    forward_es = self.forward_layers[0](lattice)
    rev_backward_es = self.backward_layers[0](lattice.reversed())

    for layer_i in range(1, len(self.forward_layers)):
      concat_fwd = Lattice(nodes=[node_fwd.new_node_with_val(dy.concatenate([node_fwd.value,node_bwd.value])) for node_fwd,node_bwd in zip(forward_es,reversed(rev_backward_es.nodes))])
      concat_bwd = Lattice(nodes=[node_bwd.new_node_with_val(dy.concatenate([node_fwd.value,node_bwd.value])) for node_fwd,node_bwd in zip(reversed(forward_es.nodes),rev_backward_es)])
      new_forward_es = self.forward_layers[layer_i](concat_fwd)
      rev_backward_es = self.backward_layers[layer_i](concat_bwd)
      forward_es = new_forward_es

    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
                          for layer_i in range(len(self.forward_layers))]
    return Lattice(nodes=[lattice.nodes[i].new_node_with_val(dy.concatenate([forward_es[i].value,rev_backward_es[-i-1].value])) for i in range(len(forward_es))])
