import io
import random

import dynet as dy

from xnmt.transducer import Transducer, FinalTransducerState
from xnmt.input import Input, BaseTextReader
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
from xnmt.vocab import Vocab
from xnmt.embedder import SimpleWordEmbedder
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.batcher import is_batched
from xnmt.expression_sequence import ExpressionSequence

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
  def __init__(self, nodes):
    """
    :param nodes: list of LatticeNode objects
    """
    self.nodes = nodes
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
  def _add_bwd_connections(self, nodes):
    for pos in range(len(nodes)):
      for pred_i in nodes[pos].nodes_prev:
        nodes[pred_i].nodes_next.append(pos)
    return nodes
 
  def reversed(self):
    rev_nodes = []
    seq_len = len(self.nodes)
    for node in reversed(self.nodes):
      new_node = LatticeNode(nodes_prev = [seq_len - n - 1 for n in node.nodes_next],
                             nodes_next = [seq_len - p - 1 for p in node.nodes_prev],
                             value = node.value)
      rev_nodes.append(new_node)
    return Lattice(rev_nodes)
  
class BinnedLattice(Lattice):
  def __init__(self, bins):
    """
    :param bins: indexes bins[bin_pos][rep_pos][token_pos]
    """
    super(BinnedLattice, self).__init__(nodes = self.bins_to_nodes(bins))
    self.bins = bins
  
  def __repr__(self):
    return str(self.bins)
  
  def bins_to_nodes(self, bins, drop_arcs=0.0):
    assert len(bins[0]) == len(bins[-1]) == len(bins[0][0]) == len(bins[-1][0]) == 1
    nodes = [LatticeNode([], [], bins[0][0][0])]
    prev_indices = [0]
    for cur_bin in bins[1:-1]:
      new_prev_indices = []
      if drop_arcs > 0.0:
        shuffled_bin = list(cur_bin)
        random.shuffle(shuffled_bin)
        dropped_bin = [shuffled_bin[0]]
        for b in shuffled_bin[1:]:
          if random.random() > drop_arcs:
            dropped_bin.append(b)
        cur_bin = dropped_bin
      for rep in cur_bin:
        for rep_pos in range(len(rep)):
          if rep_pos==0:
            preds = prev_indices
          else:
            preds = [len(nodes)-1]
          #print("node", len(nodes), preds)
          nodes.append(LatticeNode(preds, [], rep[rep_pos]))
        new_prev_indices.append(len(nodes)-1)
      prev_indices = new_prev_indices
    nodes.append(LatticeNode(prev_indices, [], bins[-1][0][0]))
    return self._add_bwd_connections(nodes)
  
  def drop_arcs(self, dropout):
    return Lattice(nodes=self.bins_to_nodes(self.bins, drop_arcs=dropout))
  

  
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
      bins = []
      for word in words:
        representations = self.get_representations(word)
        cur_bin = []
        for rep in representations:
          cur_rep_mapped = []
          for rep_token in rep:
            cur_rep_mapped.append(self.vocab.convert(rep_token))
          cur_bin.append(cur_rep_mapped)
        bins.append(cur_bin)
      lattice = BinnedLattice(bins=bins)
      sents.append(lattice)
    return sents
  


  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param(u"vocab", self.vocab)

  def vocab_size(self):
    return len(self.vocab)
  
  def get_representations(self, word):
    if word in [Vocab.ES_STR, Vocab.SS_STR, Vocab.UNK_STR]:
      return [[word]]
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

  def __init__(self, vocab = None, vocab_size = None, emb_dim = None, word_dropout = 0.0,
               arc_dropout = 0.0, xnmt_global=Ref(Path("xnmt_global")),
               yaml_path = None,
               src_reader = Ref(path=Path("model.src_reader"), required=False),
               trg_reader = Ref(path=Path("model.trg_reader"), required=False)):
    """
    :param vocab_size:
    :param emb_dim:
    :param word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    """
    register_handler(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.emb_dim = emb_dim or xnmt_global.default_layer_dim
    self.word_dropout = word_dropout
    self.embeddings = xnmt_global.dynet_param_collection.param_col.add_lookup_parameters((self.vocab_size, self.emb_dim))
    self.word_id_mask = None
    self.weight_noise = 0.0
    self.fix_norm = None
    self.arc_dropout = arc_dropout

  def embed_sent(self, sent):
    if is_batched(sent):
      assert len(sent)==1, "LatticeEmbedder requires batch size of 1"
      assert sent.mask is None
      sent = sent[0]
    if self.train and self.arc_dropout > 0.0:
      sent = sent.drop_arcs(self.arc_dropout)
    embedded_nodes = [word.new_node_with_val(self.embed(word.value)) for word in sent]
    return Lattice(nodes=embedded_nodes)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val


class LatticeLSTMTransducer(Transducer):
  def __init__(self, input_dim, hidden_dim, dropout = 0.0, xnmt_global=Ref(Path("xnmt_global"))):
    register_handler(self)
    self.dropout_rate = dropout or xnmt_global.dropout
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = xnmt_global.dynet_param_collection.param_col

    # [i; o; g]
    self.p_Wx_iog = model.add_parameters(dim=(hidden_dim*3, input_dim))
    self.p_Wh_iog = model.add_parameters(dim=(hidden_dim*3, hidden_dim))
    self.p_b_iog  = model.add_parameters(dim=(hidden_dim*3,), init=dy.ConstInitializer(0.0))
    self.p_Wx_f = model.add_parameters(dim=(hidden_dim, input_dim))
    self.p_Wh_f = model.add_parameters(dim=(hidden_dim, hidden_dim))
    self.p_b_f  = model.add_parameters(dim=(hidden_dim,), init=dy.ConstInitializer(1.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
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

  def __call__(self, lattice):
    Wx_iog = dy.parameter(self.p_Wx_iog)
    Wh_iog = dy.parameter(self.p_Wh_iog)
    b_iog = dy.parameter(self.p_b_iog)
    Wx_f = dy.parameter(self.p_Wx_f)
    Wh_f = dy.parameter(self.p_Wh_f)
    b_f = dy.parameter(self.p_b_f)
    h = []
    c = []

    batch_size = lattice[0].value.dim()[1]
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    for x_t in lattice:
      val = x_t.value
      if self.dropout_rate > 0.0 and self.train:
        val = dy.cmult(val, self.dropout_mask_x)
      i_ft_list = []
      if len(x_t.nodes_prev)==0:
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val])
      else:
        h_tilde = sum(h[pred] for pred in x_t.nodes_prev)
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val, Wh_iog, h_tilde])
        for pred in x_t.nodes_prev:
          i_ft_list.append(dy.logistic(dy.affine_transform([b_f, Wx_f, val, Wh_f, h[pred]])))
      i_ait = dy.pick_range(tmp_iog, 0, self.hidden_dim)
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
      h_t = dy.cmult(i_ot, dy.tanh(c[-1]))
      if self.dropout_rate > 0.0 and self.train:
        h_t = dy.cmult(h_t, self.dropout_mask_h)
      h.append(h_t)
    self._final_states = [FinalTransducerState(h[-1], c[-1])]
    return Lattice(nodes=[node_t.new_node_with_val(h_t) for node_t, h_t in zip(lattice.nodes,h)])

class BiLatticeLSTMTransducer(Transducer, Serializable):
  yaml_tag = u'!BiLatticeLSTMTransducer'
  
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), layers=1, input_dim=None, hidden_dim=None, dropout=None):
    register_handler(self)
    self.num_layers = layers
    input_dim = input_dim or xnmt_global.default_layer_dim
    hidden_dim = hidden_dim or xnmt_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or xnmt_global.dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = [LatticeLSTMTransducer(xnmt_global=xnmt_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout)]
    self.backward_layers = [LatticeLSTMTransducer(xnmt_global=xnmt_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout)]
    self.forward_layers += [LatticeLSTMTransducer(xnmt_global=xnmt_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout) for _ in range(layers-1)]
    self.backward_layers += [LatticeLSTMTransducer(xnmt_global=xnmt_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout) for _ in range(layers-1)]

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, lattice):
    if isinstance(lattice, ExpressionSequence):
      lattice = Lattice([LatticeNode([i-1] if i>0 else [], [i+1] if i<len(lattice)-1 else [], value) for (i,value) in enumerate(lattice)])
    
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
