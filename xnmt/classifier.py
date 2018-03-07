from simple_settings import settings
import numpy as np
import dynet as dy

from xnmt.batcher import is_batched, mark_as_batch
from xnmt.embedder import SimpleWordEmbedder
from xnmt.generator import GeneratorModel
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.linear import Linear
from xnmt.output import TextOutput
from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.tree_tools import Path, Ref


class ClassifierInference(Serializable):
  yaml_tag = "!ClassifierInference"
  
  def __init__(self, batcher=Ref(Path("train.batcher"), required=False)):
    self.batcher = batcher
  
  def __call__(self, generator, src_file=None, trg_file=None, candidate_id_file=None):
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Perform generation of output
    with open(trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
      src_ret=[]
      for i, src in enumerate(src_corpus):
        # This is necessary when the batcher does some sort of pre-processing, e.g.
        # when the batcher pads to a particular number of dimensions
        if self.batcher:
          self.batcher.add_single_batch(src_curr=[src], trg_curr=None, src_ret=src_ret, trg_ret=None)
          src = src_ret.pop()[0]
  
        # Do the decoding
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        output = generator.generate_output(src, i)
        output_txt = output[0].actions
        # Printing to trg file
        fp.write(f"{output_txt}\n")  

class Classifier(GeneratorModel):
  '''
  A template class implementing a sequence classifier that can calculate a
  loss and generate the output class.
  '''

  def calc_loss(self, src, cls):
    '''Calculate loss based on input-output pairs.

    :param src: The source, a sentence or a batch of sentences.
    :param trg: The target, a class label or a batch of class labels.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Classifier subclasses')
  def get_primary_loss(self):
    return "standard_loss"

class DefaultClassifier(Classifier, Serializable):
  
  yaml_tag = "!DefaultClassifier"
  
  def __init__(self, classifier_input_dim, num_classes, src_reader, trg_reader,
               src_embedder=bare(SimpleWordEmbedder),
               encoder=bare(BiLSTMSeqTransducer),
               inference=bare(ClassifierInference),
               exp_global=Ref(Path("exp_global"))):
    '''Constructor.

    :param src_reader: A reader for the source side.
    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_reader: A reader for the target side.
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    :param inference: The default inference strategy used for this model
    '''
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.inference = inference
    self.softmax_out = Linear(input_dim = classifier_input_dim,
                           output_dim = num_classes,
                           model = exp_global.dynet_param_collection.param_col)

  def shared_params(self):
    return [set([Path(".src_embedder.emb_dim"), Path(".encoder.input_dim")]),
            set([Path(".encoder.hidden_dim"), Path(".classifier_input_dim")])]
  
  def calc_loss(self, src, cls, loss_calculator):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param cls: target class label (unbatched or batched)
    :returns: (possibly batched) loss expression
    """
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    self.encoder(embeddings)
    scores = self.softmax_out(self.encoder.get_final_states()[-1].main_expr())
    if not is_batched(cls):
      return dy.pickneglogsoftmax(scores, cls)
    else:
      return dy.pickneglogsoftmax_batch(scores, cls)
  
  def generate(self, src, idx, src_mask=None, forced_trg_ids=None):
    if not is_batched(src):
      src = mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    for sents in src:
      self.start_sent(sents)
      embeddings = self.src_embedder.embed_sent(sents)
      self.encoder(embeddings)
      scores = self.softmax_out(self.encoder.get_final_states()[-1].main_expr())
      logsoftmax = dy.log_softmax(scores).npvalue()
      output_actions = np.argmax(logsoftmax)
      score = np.max(logsoftmax)
      # Append output to the outputs
      outputs.append(TextOutput(actions=output_actions,
                                vocab=None,
                                score=score))
    self.outputs = outputs
    return outputs
