from __future__ import division, generators

import six
import math
import random
import numpy as np
import dynet as dy
from xnmt.vocab import Vocab
from xnmt.serializer import Serializable

class Batch(list):
  """
  Specialization of list that indicates a (mini)batch of things, together with an optional mask.
  Should be treated as immutable object.
  """
  def __init__(self, batch_list, mask=None):
    super(Batch, self).__init__(batch_list)
    self.mask = mask

class Mask(object):
  """
  Masks are represented as numpy array of dimensions batchsize x seq_len, with parts
  belonging to the sequence set to 0, and parts that should be masked set to 1
  """
  def __init__(self, np_arr):
    assert len(np_arr.shape)==2
    self.np_arr = np_arr
  
  def __len__(self):
    return self.np_arr.shape[1]
 
  def batch_size(self):
    return self.np_arr.shape[0]

  def reversed(self):
    return Mask(self.np_arr[:,::-1])
  
  def lin_subsampled(self, reduce_factor=None, trg_len=None):
    if reduce_factor:
      return Mask(np.array([[self.np_arr[b,int(i*reduce_factor)] for i in range(int(math.ceil(len(self)/float(reduce_factor))))] for b in range(self.batch_size())]))
    else:
      return Mask(np.array([[self.np_arr[b,int(i*len(self)/float(trg_len))] for i in range(trg_len)] for b in range(self.batch_size())]))
      
  
  def add_to_tensor_expr(self, tensor_expr, multiplicator=None, time_first=False):
    assert tensor_expr.dim()[1]>1 or self.np_arr.shape[0]==1
    # TODO: might cache these expressions to save memory
    if np.count_nonzero(self.np_arr) == 0:
      return tensor_expr
    else:
      dim_before = tensor_expr.dim()
      # fill dims up with 1's in the beginning so that number of dims match:
      reshape_size = self.mask_reshape_size(tensor_expr.dim(), time_first)
      if multiplicator is not None:
        mask_expr = dy.inputTensor(np.reshape(self.np_arr.transpose(), reshape_size) * multiplicator, batched=True)
      else:
        mask_expr = dy.inputTensor(np.reshape(self.np_arr.transpose(), reshape_size), batched=True)
      ret = tensor_expr + mask_expr
      assert dim_before == ret.dim()
      return ret

  def broadcast_factor(self, tensor_expr):
    """
    returns product(tensor_expr dims) / product(mask dims)
    """
    tensor_expr_size = tensor_expr.dim()[1]
    for d in tensor_expr.dim()[0]: tensor_expr_size *= d
    return tensor_expr_size / self.np_arr.size
  
  def mask_reshape_size(self, tensor_dim, time_first=False):
    if time_first:
      return list(reversed(self.np_arr.shape[1:])) + [1] * (len(tensor_dim[0]) - len(self.np_arr.shape) + 1) + [self.np_arr.shape[0]]
    else:
      return [1] * (len(tensor_dim[0]) - len(self.np_arr.shape) + 1) + list(reversed(self.np_arr.shape))

  def set_masked_to_mean(self, tensor_expr, time_first=False):
    """
    Set masked parts of the tensor expr to the mean of the unmasked parts.
    """
    if np.count_nonzero(self.np_arr) == 0:
      return tensor_expr
    else:
      dim_before = tensor_expr.dim()
      reshape_size = self.mask_reshape_size(tensor_expr.dim(), time_first)
      inv_mask_expr = dy.inputTensor(1.0 - np.reshape(self.np_arr.transpose(), reshape_size), batched=True)
      unmasked = dy.cmult(tensor_expr, inv_mask_expr)
      unmasked_mean = unmasked
      while sum(unmasked_mean.dim()[0]) > 1: # loop because mean_dim only supports reducing up to 2 dimensions at a time
        unmasked_mean = dy.mean_dim(unmasked_mean, list(range(min(2,len(unmasked_mean.dim()[0])))), unmasked_mean.dim()[1]>1, n=1) # this is mean without normalization == sum
      unmasked_mean = dy.cdiv(unmasked_mean, dy.inputTensor(np.asarray([(self.np_arr.size - np.count_nonzero(self.np_arr)) * self.broadcast_factor(tensor_expr)]), batched=False))
      mask_expr = dy.cmult(dy.inputTensor(np.reshape(self.np_arr.transpose(), reshape_size), batched=True), unmasked_mean)
      ret = unmasked + mask_expr
      assert ret.dim() == dim_before
      return ret

  def cmult_by_timestep_expr(self, expr, timestep, inverse=False):
    # TODO: might cache these expressions to save memory
    """
    :param expr: a dynet expression corresponding to one timestep
    :param timestep: index of current timestep
    :param inverse: True will keep the unmasked parts, False will zero out the unmasked parts
    """
    if inverse:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == 0:
        return expr
      mask_exp = dy.inputTensor((1.0 - self.np_arr)[:,timestep:timestep+1].transpose(), batched=True)
    else:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == self.np_arr[:,timestep:timestep+1].size:
        return expr
      mask_exp = dy.inputTensor(self.np_arr[:,timestep:timestep+1].transpose(), batched=True)
    return dy.cmult(expr, mask_exp)

class Batcher(object):
  """
  A template class to convert a list of sents to several batches of sents.
  """

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    self.batch_size = batch_size
    self.src_pad_token = src_pad_token
    self.trg_pad_token = trg_pad_token
    self.granularity = granularity

  def is_random(self):
    """
    :returns: True if there is some randomness in the batching process, False otherwise. Defaults to false.
    """
    return False

  def add_single_batch(self, src_curr, trg_curr, src_ret, trg_ret):
    src_id, src_mask = pad(src_curr, pad_token=self.src_pad_token)
    src_ret.append(Batch(src_id, src_mask))
    trg_id, trg_mask = pad(trg_curr, pad_token=self.trg_pad_token)
    trg_ret.append(Batch(trg_id, trg_mask))

  def pack_by_order(self, src, trg, order):
    src_ret, src_curr = [], []
    trg_ret, trg_curr = [], []
    if self.granularity == 'sent':
      for x in six.moves.range(0, len(order), self.batch_size):
        self.add_single_batch([src[y] for y in order[x:x+self.batch_size]], [trg[y] for y in order[x:x+self.batch_size]], src_ret, trg_ret)
    elif self.granularity == 'word':
      my_size = 0
      for i in order:
        my_size += len_or_zero(src[i]) + len_or_zero(trg[i])
        if my_size > self.batch_size and len(src_curr)>0:
          self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret)
          my_size = len(src[i]) + len(trg[i])
          src_curr = []
          trg_curr = []
        src_curr.append(src[i])
        trg_curr.append(trg[i])
      self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret)
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    return src_ret, trg_ret

class InOrderBatcher(Batcher, Serializable):
  yaml_tag = u"!InOrderBatcher"
  """
  A class to create batches in order of the original corpus.
  """
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    super(InOrderBatcher, self).__init__(batch_size, src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)

  def pack(self, src, trg):
    order = list(range(len(src)))
    return self.pack_by_order(src, trg, order)

class ShuffleBatcher(Batcher):
  """
  A class to create batches through randomly shuffling without sorting.
  """

  def pack(self, src, trg):
    order = list(range(len(src)))
    np.random.shuffle(order)
    return self.pack_by_order(src, trg, order)

  def is_random(self):
    return True

class SortBatcher(Batcher):
  """
  A template class to create batches through bucketing sent length.
  """
  __tiebreaker_eps = 1.0e-7

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES,
               trg_pad_token=Vocab.ES, sort_key=lambda x: len(x[0]),
               break_ties_randomly=True):
    super(SortBatcher, self).__init__(batch_size, granularity=granularity,
                                      src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    self.sort_key = sort_key
    self.break_ties_randomly = break_ties_randomly

  def pack(self, src, trg):
    if self.break_ties_randomly:
      order = np.argsort([self.sort_key(x) + random.uniform(-SortBatcher.__tiebreaker_eps, SortBatcher.__tiebreaker_eps) for x in six.moves.zip(src,trg)])
    else:
      order = np.argsort([self.sort_key(x) for x in six.moves.zip(src,trg)])
    return self.pack_by_order(src, trg, order)

  def is_random(self):
    return self.break_ties_randomly

# Module level functions
def mark_as_batch(data, mask=None):
  if type(data) == Batch and mask is None:
    ret = data
  else:
    ret = Batch(data, mask)
  return ret

def is_batched(data):
  return type(data) == Batch

def pad(batch, pad_token=Vocab.ES):
  max_len = max(len_or_zero(item) for item in batch)
  min_len = min(len_or_zero(item) for item in batch)
  if min_len == max_len:
    return batch, None
  masks = np.zeros([len(batch), max_len])
  for i, v in enumerate(batch):
    for j in range(len_or_zero(v), max_len):
      masks[i,j] = 1.0
  padded_items = [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch]
  return padded_items, Mask(masks)

def len_or_zero(val):
  return len(val) if hasattr(val, '__len__') else 0

class SrcBatcher(SortBatcher, Serializable):
  yaml_tag = u"!SrcBatcher"
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(SrcBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[0]), granularity='sent',
                                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                     break_ties_randomly=break_ties_randomly)

class TrgBatcher(SortBatcher, Serializable):
  yaml_tag = u"!TrgBatcher"
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(TrgBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[1]), granularity='sent',
                                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                     break_ties_randomly=break_ties_randomly)

class SrcTrgBatcher(SortBatcher, Serializable):
  yaml_tag = u"!SrcTrgBatcher"
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(SrcTrgBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]),
                                        granularity='sent',
                                        src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                        break_ties_randomly=break_ties_randomly)

class TrgSrcBatcher(SortBatcher, Serializable):
  yaml_tag = u"!TrgSrcBatcher"
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(TrgSrcBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]),
                                        granularity='sent',
                                        src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                        break_ties_randomly=break_ties_randomly)

class SentShuffleBatcher(ShuffleBatcher, Serializable):
  yaml_tag = u"!SentShuffleBatcher"

  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    super(SentShuffleBatcher, self).__init__(batch_size, granularity='sent', src_pad_token=src_pad_token,
                                             trg_pad_token=trg_pad_token)

class WordShuffleBatcher(ShuffleBatcher, Serializable):
  yaml_tag = u"!WordShuffleBatcher"

  def __init__(self, words_per_batch, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    super(WordShuffleBatcher, self).__init__(words_per_batch, granularity='word', src_pad_token=src_pad_token,
                                             trg_pad_token=trg_pad_token)

class WordSortBatcher(SortBatcher, Serializable):
  def __init__(self, words_per_batch, avg_batch_size, sort_key,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    # Sanity checks
    if words_per_batch and avg_batch_size:
      raise ValueError("words_per_batch and avg_batch_size are mutually exclusive.")
    elif words_per_batch is None and avg_batch_size is None:
      raise ValueError("either words_per_batch or avg_batch_size must be specified.")

    super(WordSortBatcher, self).__init__(words_per_batch, sort_key=sort_key, granularity='word',
                                          src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                          break_ties_randomly=break_ties_randomly)
    self.avg_batch_size = avg_batch_size

class WordSrcBatcher(WordSortBatcher, Serializable):
  yaml_tag = u"!WordSrcBatcher"

  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(WordSrcBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[0]),
                                         src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                         break_ties_randomly=break_ties_randomly)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = sum([len(s) for s in src]) / len(src) * self.avg_batch_size
    return super(WordSrcBatcher, self).pack_by_order(src, trg, order)

class WordTrgBatcher(WordSortBatcher, Serializable):
  yaml_tag = u"!WordTrgBatcher"

  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(WordTrgBatcher, self).__init__(words_per_batch, sort_key=lambda x: len(x[1]), granularity='word',
                                         src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                         break_ties_randomly=break_ties_randomly)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = sum([len(s) for s in trg]) / len(trg) * self.avg_batch_size
    return super(WordTrgBatcher, self).pack_by_order(src, trg, order)

class WordSrcTrgBatcher(WordSortBatcher, Serializable):
  yaml_tag = u"!WordSrcTrgBatcher"

  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    super(WordSrcTrgBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]),
                                            src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                            break_ties_randomly=break_ties_randomly)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = sum([len(s) for s in src]) / len(src) * self.avg_batch_size
    return super(WordSrcTrgBatcher, self).pack_by_order(src, trg, order)

class WordTrgSrcBatcher(WordSortBatcher, Serializable):
  yaml_tag = u"!WordTrgSrcBatcher"

  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True):
    return super(WordTrgSrcBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]),
                                            src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                            break_ties_randomly=break_ties_randomly)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = sum([len(s) for s in trg]) / len(trg) * self.avg_batch_size
    return super(WordTrgSrcBatcher, self).pack_by_order(src, trg, order)

