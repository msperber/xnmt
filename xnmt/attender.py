import dynet as dy
from xnmt.serializer import Serializable
from xnmt.hier_model import HierarchicalModel, recursive

class Attender(object):
  '''
  A template class for functions implementing attention.
  '''

  def __init__(self, input_dim):
    """
    :param input_dim: every attender needs an input_dim
    """
    pass

  def init_sent(self, sent):
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, state):
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')


class StandardAttender(Attender, Serializable, HierarchicalModel):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  yaml_tag = u'!StandardAttender'

  def __init__(self, yaml_context, input_dim=None, state_dim=None, hidden_dim=None,
               dropout=None, dropout_scores=False):
    input_dim = input_dim or yaml_context.default_layer_dim
    state_dim = state_dim or yaml_context.default_layer_dim
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = yaml_context.dynet_param_collection.param_col
    self.pW = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb = param_collection.add_parameters(hidden_dim)
    self.pU = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent = None
    self.dropout = dropout # TODO: or yaml_context.dropout
    self.dropout_scores = dropout_scores
    self.train = False

  def init_sent(self, sent):
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)

    self.WI = dy.affine_transform([b, W, I])
    wi_dim = self.WI.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI = dy.reshape(self.WI, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0, time_first=True)
    if self.train and self.dropout > 0.0 and self.dropout_scores:
      dropout_mask = dy.random_bernoulli(scores.dim()[0], self.dropout, batch_size=scores.dim()[1])
      scores = dy.cmult(scores, dropout_mask)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    context = I * attention
    if self.train and self.dropout > 0.0 and not self.dropout_scores:
      context = dy.dropout(context, self.dropout)
    return context
  
  @recursive
  def set_train(self, val):
    self.train = val


class TransformerAttender(Attender, Serializable):
  yaml_tag = u'!TransformerAttender'

  def __init__(self, context, input_dim=None, state_dim=None, hidden_dim=None):
    input_dim = input_dim or context.default_layer_dim
    state_dim = state_dim or context.default_layer_dim
    hidden_dim = hidden_dim or context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = context.dynet_param_collection.param_col
