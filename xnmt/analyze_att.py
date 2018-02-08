from collections import  Counter

import numpy as np

import yaml

from xnmt.vocab import Vocab

vocab = Vocab(vocab_file = "/project/data-audio/tedlium-multi/parallel/vocab/en-de-es-fr.lc.no-numbers-punct.vocab") # /Users/matthias/Desktop/en-de-es-fr.lc.no-numbers-punct.vocab
nheads = 8
nlayers = 2

with open("/project/iwslt2015b/project/nmt-audio/exp-xnmt/03.audio2char/logs/exp165a.att.1.log.yaml", 'r') as f:
#with open("/Users/matthias/Desktop/exp165a.att.1.log.small.yaml", 'r') as f:
  yaml_log = yaml.load(f)
  sentence_logs = [[]]
  prev_key = None
  for entry in yaml_log:
    cur_key = entry["key"]
    if prev_key == 'forced_dec_id' and cur_key == 'selfatt_mat_ax0':
      sentence_logs.append([])
    sentence_logs[-1].append(entry)
    prev_key = cur_key
  axis0_sum, axis1_sum, axis0_cnt, axis1_cnt = Counter(), Counter(), Counter(), Counter() 
  for sent_log in sentence_logs:
    for layer_i in range(nlayers):
      for head_i in range(nheads):
        # shape layer 0: (downsampled_src_len*2,)
        # shape layer 1: (downsampled_src_len,)
        self_att_ax0 = np.loads(sent_log[layer_i*2]["value"])[:,head_i]
        self_att_ax1 = np.loads(sent_log[layer_i*2+1]["value"])[:,head_i]
        if layer_i==1:
          self_att_ax0 = np.repeat(self_att_ax0, 2)
          self_att_ax1 = np.repeat(self_att_ax1, 2)
        for token_i in range(len(vocab)):
          # cross_att_sum.shape = (downsampled_src_len, 1)
          cross_att_sum = np.sum([np.loads(sent_log[i]["value"]) for i in range(len(sent_log)-1) if sent_log[i]["key"]=="attention" and sent_log[i+1]["value"]==token_i],axis=0)
          if cross_att_sum.shape:
            cross_att_sum = np.repeat(cross_att_sum, 2)
            ca0 = np.corrcoef(self_att_ax0, cross_att_sum)[0,1]
            ca1 = np.corrcoef(self_att_ax1, cross_att_sum)[0,1]
            axis0_sum[(layer_i,head_i,vocab[token_i])] += ca0
            axis1_sum[(layer_i,head_i,vocab[token_i])] += ca1
            axis0_cnt[(layer_i,head_i,vocab[token_i])] += 1
            axis1_cnt[(layer_i,head_i,vocab[token_i])] += 1
  axis0_avg, axis1_avg = Counter(), Counter()
  axis0_avg_arr, axis1_avg_arr = np.zeros((nheads*nlayers, len(vocab))), np.zeros((nheads*nlayers, len(vocab)))
  for layer_i in range(nlayers):
    for head_i in range(nheads):
      for token_i in range(len(vocab)):
        ind = (layer_i,head_i,vocab[token_i])
        if ind in axis0_cnt: 
          axis0_avg[ind] = axis0_sum[ind] / axis0_cnt[ind]
          axis1_avg[ind] = axis1_sum[ind] / axis1_cnt[ind]
          axis0_avg_arr[nheads*layer_i+head_i,token_i] = axis0_avg[ind]
          axis1_avg_arr[nheads*layer_i+head_i,token_i] = axis1_avg[ind]
  print(axis0_avg)
  print("-----")
  print(axis1_avg)
  print("-----")
  print(axis0_avg_arr)
  print("-----")
  print(axis1_avg_arr)
  print("-----")
  with open("/home/msperber/experiments/attention-heads/165.1.yaml", "w") as f_out:
    f_out.write(yaml.dump({
    "axis0_avg":axis0_avg,
    "axis1_avg":axis1_avg,
    "axis0_avg_arr":axis0_avg_arr,
    "axis1_avg_arr":axis1_avg_arr
    }))


# clustering: check
# - https://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python
# - http://scikit-learn.org/stable/modules/biclustering.html