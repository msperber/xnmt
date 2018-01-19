from xnmt.serialize.serializable import Serializable

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.
  '''

  yaml_tag = u'!Experiment'

  def __init__(self, model_context=None, load=None, overwrite=None, preproc=None,
               model=None, train=None, evaluate=None, random_search_report=None):
    self.model_context = model_context
    self.load = load
    self.overwrite = overwrite
    self.preproc = preproc
    self.model = model
    self.train = train
    self.evaluate = evaluate
    if load:
      model_context.dynet_param_collection.load_from_data_file(f"{load}.data")
      print(f"> populated DyNet weights from {load}.data")

    if random_search_report:
      print(f"> instantiated random parameter search: {random_search_report}")

  def __call__(self, save_fct):
    eval_scores = "Not evaluated"
    eval_only = self.model_context.eval_only
    if not eval_only:
      print("> Training")
      self.train.run_training(save_fct = save_fct)
      print('reverting learned weights to best checkpoint..')
      self.model_context.dynet_param_collection.revert_to_best_model()

    evaluate_args = self.evaluate
    if evaluate_args:
      print("> Performing final evaluation")
      eval_scores = []
      for evaluator in evaluate_args:
        eval_score, eval_words = evaluator.eval()
        if type(eval_score) == list:
          eval_scores.extend(eval_score)
        else:
          eval_scores.append(eval_score)

    return eval_scores
        