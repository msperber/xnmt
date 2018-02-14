import dynet as dy
import xnmt.linear

class MLP(object):
  def __init__(self, input_dim, hidden_dim, output_dim, model, layers=1):
    self.hidden = [xnmt.linear.Linear(input_dim, hidden_dim, model)]
    for _ in range(1,layers):
      self.hidden.append(xnmt.linear.Linear(hidden_dim, hidden_dim, model))
    self.output = xnmt.linear.Linear(hidden_dim, output_dim, model)

  def __call__(self, input_expr):
    output = input_expr
    for hidden in self.hidden:
      output = dy.tanh(hidden(output))
    return self.output(output)
