import unittest

import math
import numpy as np
import dynet_config
import dynet as dy

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.pyramidal import PyramidalLSTMSeqTransducer
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.residual import ResidualLSTMSeqTransducer
from xnmt.attender import MlpAttender
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader
from xnmt.model_context import ModelContext, PersistentParamCollection
import xnmt.batcher
import xnmt.events

class TestEncoder(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                                   train_trg = "examples/data/head.en",
                                                   dev_src = "examples/data/head.ja",
                                                   dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                               trg_reader = PlainTextReader(),
                                               training_corpus = self.training_corpus)

  @xnmt.events.register_xnmt_event
  def set_train(self, val):
    pass
  @xnmt.events.register_xnmt_event
  def start_sent(self, src):
    pass

  def assert_in_out_len_equal(self, model):
    dy.renew_cg()
    self.set_train(True)
    src = self.training_corpus.train_src_data[0]
    self.start_sent(src)
    embeddings = model.src_embedder.embed_sent(src)
    encodings = model.encoder(embeddings)
    self.assertEqual(len(embeddings), len(encodings))

  def test_bi_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=BiLSTMSeqTransducer(self.model_context, layers=3),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_uni_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=UniLSTMSeqTransducer(self.model_context),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_res_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=ResidualLSTMSeqTransducer(self.model_context, layers=3),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_py_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=PyramidalLSTMSeqTransducer(self.model_context, layers=3),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.set_train(True)
    for sent_i in range(10):
      dy.renew_cg()
      src = self.training_corpus.train_src_data[sent_i]
      self.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder(embeddings)
      self.assertEqual(int(math.ceil(len(embeddings) / float(4))), len(encodings))

  def test_py_lstm_mask(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=PyramidalLSTMSeqTransducer(self.model_context, layers=1),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )

    batcher = xnmt.batcher.TrgBatcher(batch_size=3)
    train_src, _ = \
      batcher.pack(self.training_corpus.train_src_data, self.training_corpus.train_trg_data)
    
    self.set_train(True)
    for sent_i in range(3):
      dy.renew_cg()
      src = train_src[sent_i]
      self.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder(embeddings)
      if train_src[sent_i].mask is None:
        assert encodings.mask is None
      else:
        np.testing.assert_array_almost_equal(train_src[sent_i].mask.np_arr, encodings.mask.np_arr)

if __name__ == '__main__':
  unittest.main()
