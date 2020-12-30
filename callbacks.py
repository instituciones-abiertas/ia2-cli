import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
import time

# example of callback structure
def example(param="value"):
  """
  This is just a model of callback.
  """
  def example_cb(state, logger, model, optimizer):
    return state

  return example_cb

# The real callbacks

# iteration
def print_scores_on_epoch():
  """
  write the scores in the log file
  """
  def print_scores_cb(state, logger, model, optimizer):
    e = state["i"] + 1
    ner = state["history"]["ner"][-1]
    f_score = state["history"]["f_score"][-1]
    precision_score = state["history"]["precision"][-1]
    batches = state["history"]["batches"][-1]

    logger.info("......................................................................")
    logger.info(f" Epoch N° {e}/{state['epochs']} | batches processed: {batches}")
    logger.info(f"Losses rate: ner:{ner}, f1-score: {f_score}, precision: {precision_score}")

    return state

  return print_scores_cb


def save_best_model(path_best_model="", threshold=70, score="f_score"):
  """
  Save the model if the epoch score is more than the threshold
  and if current score is a new max 
  """
  def save_best_model_cb(state, logger, model, optimizer):
    # print("last f1: ", state["history"][score][-1], " | max: ", state["max_"+score])
    
    if (state["history"][score][-1] > threshold)  and  (state["history"][score][-1] > state["max_"+score]):
        e = state["i"]+1
        with model.use_params(optimizer.averages):
            model.to_disk(path_best_model)
            logger.info(f"💾 Saving model for epoch {e}")

    return state  

  return save_best_model_cb


def reduce_lr_on_plateau(step=0.001, epochs=4, diff=1, score="f_score", last_chance=True):
  """
  Whe the model is not getting better scores (plateau or decrease)
  from the selected amount of last epochs this function sets the
  learning rate decreasing it a fixed step.
  Note: Uses the average of last scores
  :param step fixed amout to decrease the learning rate
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  :param last_chance give an extra oportunity if the last epoch has a positive diff
  """
  def reduce_lr_on_plateau_cb(state, logger, model, optimizer):
    if len(state["history"][score]) > epochs and state["lr"] > step:
      delta = np.diff(state["history"][score])[-epochs:]
     
      if np.mean(delta) < diff:
        # maybe you have been getting bad scores but the last epoch shed a glimmer of hope 
        if last_chance and delta[-1] > 0:
          logger.info(f"[reduce_lr_on_plateau] Positive rate 🛫! waiting a bit more until touch learning rate")
        else:
          state["lr"] -= step
          logger.info(f"[reduce_lr_on_plateau] Not learning then reduce learn rate to {state['lr']}")

    return state

  return reduce_lr_on_plateau_cb


def early_stop(epochs=10, score="f_score", diff=5, last_chance=True):
  """
  Sets the stop value to True in state if score is not improving during the last epochs
  Note: Uses the average of last scores
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  :param last_chance give an extra oportunity if the last epoch has a positive diff
  """
  def early_stop_cb(state, logger, model, optimizer):
    if len(state["history"][score]) > epochs:
      delta = np.diff(state["history"][score])[-epochs:]
      
      print(delta, " - suma de diff: ", np.sum(delta))
      if np.sum(delta) < diff:
        # maybe you have been getting bad scores but the last epoch shed a glimmer of hope 
        if last_chance and delta[-1] > 0:
          logger.info(f"[early_stop] Positive rate 🛫! One more chance")
        else:
          state["stop"] = True
          logger.info(f"[early_stop] Not learning what I want 😭😭. Bye Bye Adieu!")

    return state

  return early_stop_cb


def update_best_scores():
  """
  Update max or min scores in state based on history
  """
  def update_best_scores_cb(state, logger, model, optimizer):
    # max and min
    state["min_ner"] = min(state["history"]["ner"])
    state["max_f_score"] = max(state["history"]["f_score"])
    state["max_recall"] = max(state["history"]["recall"])
    state["max_precision"] = max(state["history"]["precision"])
    return state

  return update_best_scores_cb

# batch plugins
def sleep(secs=0.5, log=False):
  """
  Sleep the train loop procces some secs.
  This is experimental, however has improved loop performances
  in some cases.
  """
  def sleep_cb(state, logger, model, optimizer):
    if log:
      logger.info(f"😴😴😴 sleeping for {secs} secs")
    time.sleep(secs)
    return state

  return sleep_cb

# on stop plugins

def log_best_scores():
  """
  Logs the max/mins from state
  TODO add validation scores
  """
  def log_best_scores_cb(state, logger, model, optimizer):
    logger.info("\n\n-------🏆-BEST-SCORES-🏅----------")
    e = state["i"] + 1
    logger.info(f"using a dataset of length {state['train_size']} in {e}/{state['epochs']}")
    logger.info(f"NER -> min {state['min_ner']} | val min 👷‍♀️")
    logger.info(f"RECALL -> max {state['max_recall']} | val max 👷")
    logger.info(f"PRECISION -> max {state['max_precision']} | val max 🏗")
    logger.info(f"F-SCORE -> max {state['max_f_score']} | val max 🚧")
    
    return state

  return log_best_scores_cb


def save_csv_history(file="history.csv", session=""):
  """
  Save history values to csv file
  """
  def save_csv_history_cb(state, logger, model, optimizer):
    return state

  return save_csv_history_cb