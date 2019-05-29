from typing import Sequence, Union, Optional, Any
import contextlib

import xnmt.tensor_tools as tt
import xnmt
from xnmt import batchers, event_trigger, events, inferences, input_readers, loss_calculators, losses, reports, utils, \
  xnmt_evaluate
from xnmt.eval import metrics
from xnmt.models import base as model_base
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.settings import settings
if xnmt.backend_torch:
  import torch

class EvalTask(object):
  """
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  """
  def eval(self) -> 'metrics.EvalScore':
    raise NotImplementedError("EvalTask.eval() needs to be implemented in child classes")

class LossEvalTask(EvalTask, Serializable):
  """
  A task that does evaluation of the loss function.

  Args:
    src_file: source file name
    ref_file: reference file name
    model: generator model to use for inference
    batcher: batcher to use
    loss_calculator: loss calculator
    max_src_len: omit sentences with source length greater than specified number
    max_trg_len: omit sentences with target length greater than specified number
    max_num_sents: compute loss only for the first n sentences in the given corpus
    desc: description to pass on to computed score objects
  """
  yaml_tag = '!LossEvalTask'

  @serializable_init
  def __init__(self,
               src_file: Union[str, Sequence[str]],
               ref_file: Optional[str] = None,
               model: 'model_base.GeneratorModel' = Ref("model"),
               batcher: batchers.Batcher = Ref("train.batcher", default=bare(batchers.SrcBatcher, batch_size=32)),
               loss_calculator: loss_calculators.LossCalculator = bare(loss_calculators.MLELoss),
               max_src_len: Optional[int] = None,
               max_trg_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               desc: Any = None) -> None:
    self.model = model
    self.loss_calculator = loss_calculator
    self.src_file = src_file
    self.ref_file = ref_file
    self.batcher = batcher
    self.src_data = None
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.max_num_sents = max_num_sents
    self.desc=desc

  def eval(self) -> 'metrics.EvalScore':
    """
    Perform evaluation task.

    Returns:
      Evaluated score
    """
    event_trigger.set_train(False)
    if self.src_data is None:
      self.src_data, self.ref_data, self.src_batches, self.ref_batches = \
        input_readers.read_parallel_corpus(src_reader=self.model.src_reader,
                                           trg_reader=self.model.trg_reader,
                                           src_file=self.src_file,
                                           trg_file=self.ref_file,
                                           batcher=self.batcher,
                                           max_num_sents=self.max_num_sents,
                                           max_src_len=self.max_src_len,
                                           max_trg_len=self.max_trg_len)
    loss_val = losses.FactoredLossVal()
    ref_words_cnt = 0
    for src, trg in zip(self.src_batches, self.ref_batches):
      with utils.ReportOnException({"src": src, "trg": trg, "graph": utils.print_cg_conditional}):
        tt.reset_graph()
        with torch.no_grad() if xnmt.backend_torch else contextlib.nullcontext():
          loss = self.loss_calculator.calc_loss(self.model, src, trg)

          ref_words_cnt += sum([trg_i.len_unpadded() for trg_i in trg])
          loss_val += loss.get_factored_loss_val()
      if settings.PRETEND: break

    loss_stats = {k: v/ref_words_cnt for k, v in loss_val.items()}

    self.src_data, self.trg_data, self.src_batches, self.trg_batches = None, None, None, None

    return metrics.LossScore(sum(loss_stats.values()),
                             loss_stats=loss_stats,
                             num_ref_words=ref_words_cnt,
                             desc=self.desc)

class AccuracyEvalTask(EvalTask, Serializable):
  """
  A task that does evaluation of some measure of accuracy.

  Args:
    src_file: path(s) to read source file(s) from
    ref_file: path(s) to read reference file(s) from
    hyp_file: path to write hypothesis file to
    model: generator model to generate hypothesis with
    eval_metrics: list of evaluation metrics (list of Evaluator objects or string of comma-separated shortcuts)
    inference: inference object
    perform_inference: Whether to generate the output or not. One eval task can use an already existing hyp_file
                       that was generated by the previous eval tasks.
    desc: human-readable description passed on to resulting score objects
  """

  yaml_tag = '!AccuracyEvalTask'

  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               src_file: Union[str,Sequence[str]],
               ref_file: Union[str,Sequence[str]],
               hyp_file: str,
               model: 'model_base.GeneratorModel' = Ref("model"),
               eval_metrics: Union[str, metrics.Evaluator, Sequence[metrics.Evaluator]] = "bleu",
               inference: Optional['inferences.Inference'] = None,
               perform_inference: bool = True,
               desc: Any = None) -> None:
    self.model = model
    if isinstance(eval_metrics, str):
      eval_metrics = [xnmt_evaluate.eval_shortcuts[shortcut]() for shortcut in eval_metrics.split(",")]
    elif not isinstance(eval_metrics, Sequence): eval_metrics = [eval_metrics]
    self.eval_metrics = eval_metrics
    self.src_file = src_file
    self.ref_file = ref_file
    self.hyp_file = hyp_file
    self.inference = inference or self.model.inference
    self.perform_inference = perform_inference
    self.desc = desc

  def eval(self) -> Sequence[metrics.EvalScore]:
    event_trigger.set_train(False)
    if issubclass(self.model.__class__, reports.Reportable):
      self.model.report_corpus_info({"ref_file": self.ref_file})
    if self.perform_inference:
      self.inference.perform_inference(generator=self.model,
                                       src_file=self.src_file,
                                       trg_file=self.hyp_file,
                                       ref_file=self.ref_file)
    # Evaluate
    eval_scores = xnmt_evaluate.xnmt_evaluate(hyp_file=self.hyp_file, ref_file=self.ref_file, desc=self.desc,
                                              evaluators=self.eval_metrics)

    return eval_scores

class DecodingEvalTask(EvalTask, Serializable):
  """
  A task that does performs decoding without comparing against a reference.

  Args:
    src_file: path(s) to read source file(s) from
    hyp_file: path to write hypothesis file to
    model: generator model to generate hypothesis with
    inference: inference object
  """

  yaml_tag = '!DecodingEvalTask'

  @serializable_init
  def __init__(self,
               src_file: Union[str,Sequence[str]],
               hyp_file: str,
               model: 'model_base.GeneratorModel' = Ref("model"),
               inference: Optional['inferences.Inference'] = None) -> None:

    self.model = model
    self.src_file = src_file
    self.hyp_file = hyp_file
    self.inference = inference or self.model.inference

  def eval(self) -> None:
    event_trigger.set_train(False)
    self.inference.perform_inference(generator=self.model,
                                     src_file=self.src_file,
                                     trg_file=self.hyp_file)
    return None
