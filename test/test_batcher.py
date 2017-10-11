import unittest

import numpy as np
import dynet as dy
import xnmt.batcher
import xnmt.input

class TestBatcher(unittest.TestCase):

  def test_batch_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = xnmt.batcher.from_spec("src", 3, src_pad_token=1, trg_pad_token=2)
    src, trg = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0, 1, 1], [0, 0, 1], [0, 0, 0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 2, 2], [0, 0, 0, 2], [0, 0, 0, 0]], [x.words for x in trg[1]])

  def test_batch_word_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = xnmt.batcher.from_spec("word_src", 12, src_pad_token=1, trg_pad_token=2)
    src, trg = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 1], [0, 0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[1]])
    self.assertEqual([[0, 0, 0, 0]], [x.words for x in src[2]])
    self.assertEqual([[0, 0]], [x.words for x in trg[2]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in src[3]])
    self.assertEqual([[0, 0, 0]], [x.words for x in trg[3]])
    self.assertEqual([[0, 0, 0, 0, 0, 0]], [x.words for x in src[4]])
    self.assertEqual([[0, 0, 0, 0]], [x.words for x in trg[4]])

class TestMask(unittest.TestCase):
  def test_lin_reversed(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    rev = mask.reversed()
    np.testing.assert_array_equal(rev.np_arr, [[1,1,0], [0,0,0]])
    rev_rev = rev.reversed()
    np.testing.assert_array_equal(rev_rev.np_arr, mask.np_arr)

  def test_lin_subsampled(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    lin_subsampled = mask.lin_subsampled(2)
    np.testing.assert_array_equal(lin_subsampled.np_arr, [[0,1], [0,0]])
    lin_subsampled2 = mask.lin_subsampled(trg_len=1)
    np.testing.assert_array_equal(lin_subsampled2.np_arr, [[0], [0]])

  def test_lin_subsampled(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    lin_subsampled = mask.lin_subsampled(2)
    np.testing.assert_array_equal(lin_subsampled.np_arr, [[0,1], [0,0]])
    lin_subsampled2 = mask.lin_subsampled(trg_len=1)
    np.testing.assert_array_equal(lin_subsampled2.np_arr, [[0], [0]])

  def test_add_to_tensor_expr_inv(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    tensor_expr = dy.inputTensor(np.random.normal(size=(2,4)))
    self.assertRaises(Exception, mask.add_to_tensor_expr, tensor_expr)
    
  def test_add_to_tensor_expr(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    val = np.random.normal(size=(3,2))
    tensor_expr = dy.inputTensor(val, batched=True)
    added = mask.add_to_tensor_expr(tensor_expr, multiplicator=None)
    np.testing.assert_array_almost_equal(added.npvalue(), mask.np_arr.transpose()+val)

  def test_add_to_tensor_expr_mult(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    val = np.random.normal(size=(3,2))
    tensor_expr = dy.inputTensor(val, batched=True)
    added = mask.add_to_tensor_expr(tensor_expr, multiplicator=-10)
    np.testing.assert_array_almost_equal(added.npvalue(), (mask.np_arr.transpose()*(-10))+val)

  def test_add_to_tensor_expr_broadcast(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    val = np.random.normal(size=(10, 3, 2))
    tensor_expr = dy.inputTensor(val, batched=True)
    added = mask.add_to_tensor_expr(tensor_expr, multiplicator=None)
    np.testing.assert_array_almost_equal(added.npvalue(), np.reshape(mask.np_arr.transpose(), (1,3,2))+val)

  def test_set_masked_to_mean(self):
    mask = xnmt.batcher.Mask(np.asarray([[0, 1, 1], [0, 0, 0]]))
    val = np.random.normal(size=(10, 3, 2))
    tensor_expr = dy.inputTensor(val, batched=True)
    meaned = mask.set_masked_to_mean(tensor_expr)
    mean_manual = sum(val[:,0,0] + val[:,0,1] + val[:,1,1] + val[:,2,1]) / 40.0
    self.assertAlmostEqual(sum(meaned.npvalue()[:,0,0]), sum(val[:,0,0]), 5)
    self.assertAlmostEqual(sum(meaned.npvalue()[:,0,1]), sum(val[:,0,1]), 5)
    self.assertAlmostEqual(sum(meaned.npvalue()[:,1,1]), sum(val[:,1,1]), 5)
    self.assertAlmostEqual(sum(meaned.npvalue()[:,2,1]), sum(val[:,2,1]), 5)
    self.assertAlmostEqual(meaned.npvalue()[0,1,0], mean_manual, 5)
    self.assertAlmostEqual(meaned.npvalue()[0,2,0], mean_manual, 5)
    self.assertAlmostEqual(meaned.npvalue()[4,1,0], mean_manual, 5)


if __name__ == '__main__':
  unittest.main()
