import argparse
from collections import  Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.spatial.distance
import numpy as np

import yaml

from xnmt.vocab import Vocab

parser = argparse.ArgumentParser(description='Analyze correlation between cross attention and self attention.')
parser.add_argument('--vocab', type=str, default="/Users/matthias/Desktop/en-de-es-fr.lc.no-numbers-punct.vocab")
parser.add_argument('--yaml_log', type=str, default="/Users/matthias/Desktop/165.3.cossim.yaml")
parser.add_argument('--summarize_yaml', type=str, default="/Users/matthias/Desktop/165.3.cossim.yaml")
parser.add_argument('--plot', type=str, default="/Users/matthias/Desktop/165.3.png")
parser.add_argument('--distance', type=str, default="corrcoef") # corrcoef | cosine
parser.add_argument('--do_summarize', dest='do_summarize', action='store_const',
                    const=True, default=False,)
parser.add_argument('--do_plot', dest='do_plot', action='store_const',
                    const=True, default=False,)
args = parser.parse_args()

# vocab_file = "/project/data-audio/tedlium-multi/parallel/vocab/en-de-es-fr.lc.no-numbers-punct.vocab"
# yaml_log_file = "/project/iwslt2015b/project/nmt-audio/exp-xnmt/03.audio2char/logs/exp165a.att.1.log.yaml"
# summarized_yaml_file = "/home/msperber/experiments/attention-heads/165.3.cossim.yaml"
# plot_file = ""
vocab_file = args.vocab #"/Users/matthias/Desktop/en-de-es-fr.lc.no-numbers-punct.vocab"
# yaml_log_file = "/Users/matthias/Desktop/exp165a.att.1.log.small.yaml"
yaml_log_file = args.yaml_log # "/Users/matthias/Desktop/165.3.cossim.yaml"
summarized_yaml_file = args.summarize_yaml # "/Users/matthias/Desktop/165.3.cossim.yaml"
plot_file = args.plot # "/Users/matthias/Desktop/165.3.png"

should_summarize_log = args.do_summarize
should_plot = args.do_plot

nheads = 8
nlayers = 2
vocab = Vocab(vocab_file = vocab_file)

def plot_mat(mat, filename, x_labels=[], dpi=120, fontsize=6):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
  ax.set_xticklabels(['']+x_labels, fontsize=fontsize, minor=True)
  ax.set_xticklabels(['']*20, fontsize=fontsize)
  ax.matshow(mat)
  ax.set_aspect('auto')
  fig.savefig(filename, dpi=dpi)
  fig.clf()
  plt.close('all')




if should_summarize_log:
  with open(yaml_log_file, 'r') as f:
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
            for i in range(len(sent_log)-1):
              if sent_log[i]["key"] == "attention":
                if sent_log[i]["key"] != "forced_dec_id":
                  raise ValueError("didn't find key 'forced_dec_id' after key 'attention', maybe this was not created using forced decoding?")
            cross_att_sum = np.sum([np.loads(sent_log[i]["value"]) for i in range(len(sent_log)-1) if sent_log[i]["key"]=="attention" and sent_log[i+1]["value"]==token_i],axis=0)
            if cross_att_sum.shape:
              cross_att_sum = np.repeat(cross_att_sum, 2)
              if args.distance=="corrcoef":
                ca0 = np.corrcoef(self_att_ax0, cross_att_sum)[0,1]
                ca1 = np.corrcoef(self_att_ax1, cross_att_sum)[0,1]
              elif args.distance=="cosine":
#               ca0 = np.dot(self_att_ax0, cross_att_sum)
#               ca1 = np.dot(self_att_ax1, cross_att_sum)
                ca0 = scipy.spatial.distance.cosine(self_att_ax0, cross_att_sum)
                ca1 = scipy.spatial.distance.cosine(self_att_ax1, cross_att_sum)
              else: raise ValueError("unknown distance {}".format(args.distance))
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
    with open(summarized_yaml_file, "w") as f_out:
      f_out.write(yaml.dump({
      "axis0_avg":axis0_avg,
      "axis1_avg":axis1_avg,
      "axis0_avg_arr":axis0_avg_arr,
      "axis1_avg_arr":axis1_avg_arr
      }))

if should_plot:
  with open(summarized_yaml_file, 'r') as f:
    yaml_results = yaml.load(f)
    data = yaml_results["axis0_avg_arr"]
    valid_vocab = [vocab[i] for i in range(len(vocab)) if sum(data[:,i]!=0.0)]
    valid_vocab_indices = [i for i in range(len(vocab)) if sum(data[:,i]!=0.0)]
    data_sel = data[:,[i for i in range(data.shape[1]) if sum(data[:,i]!=0.0)]] # drop vocab entries that never occurred
    plot_mat(data_sel, plot_file, valid_vocab)
     
    for head_i in range(16):
      print([vocab[v_i] for v_i in sorted(valid_vocab_indices, key=lambda i: -data[head_i,i])])
  




# clustering: check
# - https://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python
# - http://scikit-learn.org/stable/modules/biclustering.html

#   from sklearn.datasets import make_biclusters
#   from sklearn.datasets import samples_generator as sg
#   from sklearn.cluster.bicluster import SpectralCoclustering
#   from sklearn.metrics import consensus_score
#   model = SpectralCoclustering(n_clusters=5, random_state=0)
#   model.fit(yaml_results["axis1_avg_arr"])
#   fit_data = yaml_results["axis1_avg_arr"][np.argsort(model.row_labels_)]
#   fit_data = fit_data[:, np.argsort(model.column_labels_)]
#   fig = plt.figure()
#   ax = fig.add_subplot(111)
#   ax.matshow(fit_data)
#   ax.set_aspect('auto')
#   fig.savefig("/Users/matthias/Desktop/165.1.cluster.png", dpi=1200)
#   fig.clf()
#   plt.close('all')

