import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
import time
from datetime import datetime
import csv
import os

# example of callback structure
def example(param="value"):
  """
  This is just a model of callback.
  """
  def example_cb(state, logger, model, optimizer, disabled_pipes):
    return state

  return example_cb

# The real callbacks

# iteration
def print_scores_on_epoch(validation=True):
  """
  write the scores/loss in the log file
  :param validation when this value is false validation scores aren't printed
  """
  def print_scores_cb(state, logger, model, optimizer, disabled_pipes):
    e = state["i"] + 1
    ner = state["history"]["ner"][-1]
    f_score = state["history"]["f_score"][-1]
    precision_score = state["history"]["precision"][-1]
    batches = state["history"]["batches"][-1]    
    logger.info("......................................................................")
    logger.info(f" Epoch N° {e}/{state['epochs']} | batches processed: {batches}")
    logger.info(f"Scores : NER loss:{ner}, f1-score: {f_score}, precision: {precision_score}")
    
    # discard validation data
    if validation:
      val_f_score = state["history"]["val_f_score"][-1]
      val_precision_score = state["history"]["val_precision"][-1]  
      logger.info(f"Validation Losses rate: f1-score: {val_f_score}, precision: {val_precision_score}")

    return state

  return print_scores_cb


def save_best_model(path_best_model="", threshold=40, score="val_f_score", mode="max", test=False):
  """
  Save the model if the epoch score is more than the threshold
  or if current score is a new max.

  :param path_best_model where to save the model. This callback add this path to the history when saved
  :param threshold value to reach in order to save the first time
  :param score value to be considered
  :param mode gives the possibility to use scores with minimun like "ner" loss as trigger. 
  posible values "min" or "max"
  :param test boolean value used to trigger a score evaluation with test dataset
  """
  def save_best_model_cb(state, logger, model, optimizer, disabled_pipes):
    save = False
    
    if mode == "max":
      save = (state["history"][score][-1] >= threshold)  and  (state["history"][score][-1] > state["max_"+score])
    elif mode == "min":
      save = (state["history"][score][-1] <= threshold)  and  (state["history"][score][-1] < state["min_"+score])

    if save:
      e = state["i"]+1
      if test:
        # change this flag to 
        state["evaluate_test"] = True

      # add pipes for the current model
      for pipe in disabled_pipes:        
        model.add_pipe(pipe[1], before="ner")

      with model.use_params(optimizer.averages):
        
        model.to_disk(path_best_model)
        print("Saving model with the following pipes", model.pipe_names)
        logger.info(f"💾 Saving model for epoch {e}")
        state["history"]["saved"][state["i"]] = path_best_model
      
      #  disble other pipes
      pipe_exceptions = ["ner"]
      other_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]
      disabled_pipes = model.disable_pipes(*other_pipes)
      # since model is a reference for nlp, ensure the train continues only with ner
      assert len(model.pipe_names) == 1 and model.pipe_names[0] == "ner", "Model must be trained only with the ner pipe"

    return state

  return save_best_model_cb

def reduce_lr_on_plateau(step=0.001, epochs=4, diff=1, score="val_f_score", last_chance=True):
  """
  Whe the model is not getting better scores (plateau or decrease)
  from the selected amount of last epochs this function sets the
  learning rate decreasing it a fixed step.
  Note: Uses the average of last scores
  :param step fixed amout to decrease the learning rate
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  :param last_chance gives an extra oportunity if the last epoch has a positive diff
  """
  def reduce_lr_on_plateau_cb(state, logger, model, optimizer, disabled_pipes):
    if len(state["history"][score]) > epochs and state["lr"] > step:
      delta = np.diff(state["history"][score])[-epochs:]

      if np.mean(delta) < diff:
        # maybe you have been getting bad scores but the last epoch shed a glimmer of hope
        if last_chance and delta[-1] > 0:
          logger.info(f"[reduce_lr_on_plateau] Positive rate 🛫! waiting a bit more until touch learning rate")
        else:
          state["lr"] -= step
          state["epochs"] += 1  #CHECK! we add 1 epoch for each decrementation of the LR
          logger.info(f"[reduce_lr_on_plateau] Not learning then reduce learn rate to {state['lr']} and epochs to {state['epochs']}")

    return state

  return reduce_lr_on_plateau_cb


def early_stop(epochs=10, score="val_f_score", diff=5, last_chance=True):
  """
  Sets the stop value to True in state if score is not improving during the last epochs
  Note: Uses the average of last scores
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  :param last_chance gives an extra oportunity if the last epoch has a positive diff
  """
  def early_stop_cb(state, logger, model, optimizer, disabled_pipes):
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


def update_best_scores(validation=True):
  """
  Update max or min scores in state based on history
  :param validation when this value is false validation scores are excluded
  """
  def update_best_scores_cb(state, logger, model, optimizer, disabled_pipes):
    # max and min
    state["min_ner"] = min(state["history"]["ner"])
    state["max_f_score"] = max(state["history"]["f_score"])
    state["max_recall"] = max(state["history"]["recall"])
    state["max_precision"] = max(state["history"]["precision"])

    if validation:
      state["max_val_f_score"] = max(state["history"]["val_f_score"])
      state["max_val_recall"] = max(state["history"]["val_recall"])
      state["max_val_precision"] = max(state["history"]["val_precision"])
    return state

  return update_best_scores_cb

# batch plugins
def sleep(secs=0.5, log=False):
  """
  Sleep the train loop procces some secs.
  This is experimental, however has improved loop performances
  in some cases.
  """
  def sleep_cb(state, logger, model, optimizer, disabled_pipes):
    if log:
      logger.info(f"😴😴😴 sleeping for {secs} secs")
    time.sleep(secs)
    return state

  return sleep_cb


def change_dropout_fixed(step=0.01, until=0.5):
  """
  [experimental] change the dropout each epoch
  :param step amount to change per epoch
  :param until limit for dropout change  
  """
  def change_dropout_fixed_cb(state, logger, model, optimizer, disabled_pipes):

    state["dropout"]
    if  step > 0 and state["dropout"] < until:
      state["dropout"] += step
      logger.info(f"[change_dropout_fixed] touching dropout. New value {state['dropout']}")
    elif step < 0 and state["dropout"] > until:
      # negative step
      state["dropout"] += step
      logger.info(f"[change_dropout_fixed] touching dropout. New value {state['dropout']}")

    else:
      logger.info(f"[change_dropout_fixed] No more room for touching dropout")

    return state

  return change_dropout_fixed_cb


# on stop plugins

def log_best_scores(validation=True):
  """
  Logs the max/mins from state
  :param validation when this value is false validation scores are excluded
  """
  def log_best_scores_cb(state, logger, model, optimizer, disabled_pipes):
    logger.info("\n\n")
    logger.info("-------🏆-BEST-SCORES-🏅----------")
    e = state["i"]
    logger.info(f"using a dataset of length {state['train_size']} in {e}/{state['epochs']}")
    logger.info(f"elapsed time: {state['elapsed_time']} minutes")
    logger.info(f"NER loss -> min {state['min_ner']}")
    # Scores
    if validation:
      logger.info(f"RECALL -> max {state['max_recall']} | validation max {state['max_val_recall']}")
      logger.info(f"PRECISION -> max {state['max_precision']} | val max {state['max_val_precision']}")
      logger.info(f"F-SCORE -> max {state['max_f_score']} | val max {state['max_val_f_score']}")
    else:
      logger.info(f"RECALL -> max {state['max_recall']}")
      logger.info(f"PRECISION -> max {state['max_precision']}")
      logger.info(f"F-SCORE -> max {state['max_f_score']}")
    return state

  return log_best_scores_cb


def save_csv_history(filename="history.csv", session="", validation=True):
  """
  Save history values to csv file
  :param filename file where to write the csv rows
  :param session session id. If blank a date string is used in each call
  :param validation when this value is false validation scores are excluded
  """
  def save_csv_history_cb(state, logger, model, optimizer, disabled_pipes):
    path = f"history/{filename}"
    logger.info("\n\n")
    logger.info(f"[save_csv_history] 💾 Saving history in a {path} file")
    # create file if not exists
    if not os.path.exists(path):
      with open(path, 'w'): pass

    # give a session name if not given
    if session == "":
      now = datetime.now()
      s = now.strftime("%Y%m%d%H%M%S")
    else:
      s = session

    header = ["session", "epoch", "batches", "lr", "dropout", "ner", "f_score", "recall", "precision", "per_type_score",
     "val_f_score", "val_recall", "val_precision", "val_per_type_score"]
    rows = []

    #prepare the rows
    logger.info(f"\n[save_csv_history] preparing rows ...")
    for i in range(len(state["history"]["ner"])):

      if validation:
        val_f_score = state["history"]["val_f_score"][i]
        val_recall = state["history"]["val_recall"][i]
        val_precision = state["history"]["val_precision"][i]
        val_per_type_score = state["history"]["val_per_type_score"][i]
      else:
        val_f_score, val_recall, val_precision, val_per_type_score = None, None, None, None

      rows.append({
        "session": s,
        "epoch": i+1,
        "batches": state["history"]["batches"][i],
        "lr": state["history"]["lr"][i],
        "dropout": state["history"]["dropout"][i],
        "ner": state["history"]["ner"][i],
        "f_score": state["history"]["f_score"][i],
        "recall": state["history"]["recall"][i],
        "precision": state["history"]["precision"][i],
        "per_type_score": state["history"]["per_type_score"][i],
        "val_f_score": val_f_score,
        "val_recall": val_recall,
        "val_precision": val_precision,
        "val_per_type_score": val_per_type_score,
      })

    # opening the csv file in 'w' mode
    file = open(path, 'w', newline ='')
    with file:
      writer = csv.DictWriter(file, fieldnames = header)
      writer.writeheader()
      for r in rows:
        writer.writerow(r)

    logger.info(f"[save_csv_history] 💾 saved!! {path} file")

    return state

  return save_csv_history_cb
