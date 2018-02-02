import dynet as dy

from xnmt.initializer import LeCunUniform
from xnmt.events import register_handler, handle_xnmt_event

class Linear(object):
  def __init__(self, input_dim, output_dim, model, bias=True, init=None, glorot_gain=1.0):
    register_handler(self)
    self.bias = bias
    self.output_dim = output_dim

    if init == 'LeCunUniform':
      init_w = LeCunUniform(input_dim)
      init_b = LeCunUniform(output_dim)
    else:
      init_b = dy.ConstInitializer(0.0)
      init_w = dy.GlorotInitializer(gain=glorot_gain)

    self.W1 = model.add_parameters((output_dim, input_dim), init=init_w)
    if self.bias:
      self.b1 = model.add_parameters(output_dim, init=init_b)

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
    else:
      b1 = dy.zeros(self.output_dim)

    output = dy.affine_transform([b1, W1, input_expr])
    self.last_output.append(output)
    return output

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.last_output = []

  @handle_xnmt_event
  def on_collect_recent_outputs(self):
    return [(self, self.last_output)]
