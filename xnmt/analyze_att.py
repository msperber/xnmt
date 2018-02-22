import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.spatial.distance
import scipy.stats
import numpy as np

import yaml

from xnmt.vocab import Vocab

parser = argparse.ArgumentParser(description='Analyze correlation between cross attention and self attention.')
parser.add_argument('--vocab', type=str, default="/Users/matthias/Desktop/en-de-es-fr.lc.no-numbers-punct.vocab")
parser.add_argument('--yaml_log', type=str, default="/Users/matthias/Desktop/165.3.cossim.yaml")
parser.add_argument('--summarize_yaml', type=str, default="/Users/matthias/Desktop/165.3.cossim.yaml")
parser.add_argument('--plot', type=str, default="/Users/matthias/Desktop/165.3.png")
parser.add_argument('--distance', type=str, default="-corrcoef") # corrcoef | corrcoef2 | cosine | intersection | urelent | urelent2 | spearman
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
  cax = ax.matshow(mat)
  fig.colorbar(cax)
  ax.set_aspect('auto')
  fig.savefig(filename, dpi=dpi)
  fig.clf()
  plt.close('all')


def dist(a, b, metric):
  if metric=="corrcoef":
    return -np.corrcoef(a, b)[0,1]
  if metric=="corrcoef2":
    return -np.corrcoef(b, a)[0,1]
  elif metric=="cosine":
    return scipy.spatial.distance.cosine(a, b)
  elif metric=="intersection":
    return np.sum(np.minimum(a, b))
  elif metric=="urelent":
    return np.sum(np.multiply(a, np.log(np.divide(a, b)) + b - a))
  elif metric=="urelent2":
    return np.sum(np.multiply(b, np.log(np.divide(b, a)) + a - b))
  elif metric=="spearman":
    return -scipy.stats.spearmanr(a, b).correlation
  else: raise ValueError("unknown metric {}".format(metric))

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
      
    axis0_stats, axis1_stats = {}, {}
    for layer_i in range(nlayers):
      for head_i in range(nheads):
        for token_i in range(len(vocab)):
          for sent_log in sentence_logs:
            axis0_concat, axis1_concat, cross_att_sum_concat = None, None, None 
            # shape layer 0: (downsampled_src_len*2,)
            # shape layer 1: (downsampled_src_len,)
            self_att_ax0 = np.loads(sent_log[layer_i*2]["value"])[:,head_i]
            self_att_ax1 = np.loads(sent_log[layer_i*2+1]["value"])[:,head_i]
            if layer_i==1:
              self_att_ax0 = np.repeat(self_att_ax0, 2)
              self_att_ax1 = np.repeat(self_att_ax1, 2)
            # cross_att_sum.shape = (downsampled_src_len, 1)
            for i in range(len(sent_log)-1):
              if sent_log[i]["key"] == "attention":
                if sent_log[i+1]["key"] != "forced_dec_id":
                  raise ValueError("didn't find key 'forced_dec_id' after key 'attention', maybe this was not created using forced decoding?")
            #cross_att_sum = np.sum([np.loads(sent_log[i]["value"]) for i in range(len(sent_log)-1) if sent_log[i]["key"]=="attention" and sent_log[i+1]["value"]==token_i],axis=0)
            cross_att_sum = None
            for sent_pos in range(len(sent_log)-1):
              if sent_log[sent_pos]["key"]=="attention" and sent_log[sent_pos+1]["value"]==token_i:
                pos_att = np.loads(sent_log[sent_pos]["value"])
                pos_att_smoothed = np.zeros(pos_att.shape)
                mu = np.argmax(pos_att, axis=0)
                sig = 5.0
                pos_att_smoothed = np.asarray([(1.0/(math.sqrt(2*math.pi*sig**2)))*math.exp(-((i-mu)**2)/(2*sig**2)) for i in range(pos_att.shape[0])])
                if cross_att_sum is None:
                  cross_att_sum = pos_att_smoothed
                else:
                  cross_att_sum += pos_att_smoothed
            if cross_att_sum is not None and cross_att_sum.shape:
              cross_att_sum = np.repeat(cross_att_sum, 2)
              #plot_mat(np.reshape(cross_att_sum, (1,cross_att_sum.shape[0])), plot_file + "." + vocab[token_i].replace("/","_") + ".png")
              if axis0_concat:
                axis0_concat = np.concatenate((axis0_concat, self_att_ax0))
                axis1_concat = np.concatenate((axis1_concat, self_att_ax1))
                cross_att_sum_concat = np.concatenate((cross_att_sum_concat, cross_att_sum))
              else:
                axis0_concat = self_att_ax0
                axis1_concat = self_att_ax1
                cross_att_sum_concat = cross_att_sum
          if axis0_concat is not None:
            #plot_mat(np.reshape(axis0_concat, (1,axis0_concat.shape[0])), plot_file + ".head" + str(layer_i) + str(head_i) + ".png")
            ca0 = dist(axis0_concat, cross_att_sum_concat, args.distance)
            axis0_stats[(layer_i,head_i,vocab[token_i])] = float(ca0)
            ca1 = dist(axis1_concat, cross_att_sum_concat, args.distance)
            axis1_stats[(layer_i,head_i,vocab[token_i])] = float(ca1)
    print(axis0_stats)
    print("-----")
    print(axis1_stats)
    with open(summarized_yaml_file, "w") as f_out:
      f_out.write(yaml.dump({
      "axis0_stats":axis0_stats,
      "axis1_stats":axis1_stats,
      }))

if should_plot:
  with open(summarized_yaml_file, 'r') as f:
    yaml_results = yaml.load(f)
    data = yaml_results["axis0_stats"]
    valid_vocab = [vocab[k] for k in range(len(vocab)) if any((i,j,vocab[k]) in data for i in range(nlayers) for j in range(nheads))]
    valid_vocab_indices = [k for k in range(len(vocab)) if any((i,j,vocab[k]) in data for i in range(nlayers) for j in range(nheads))]
    mat = np.zeros((nlayers*nheads, len(valid_vocab)))
    for i in range(nlayers):
      for j in range(nheads):
        for v in valid_vocab:
          mat[nheads*i+j,valid_vocab.index(v)] = -data[(i,j,v)]
    plot_mat(mat, plot_file, valid_vocab)
     
    for i in range(nlayers):
      for j in range(nheads):
        print([v for v in sorted(valid_vocab, key=lambda k: data[(i,j,k)])])
  




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

