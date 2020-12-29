import numpy as np

# example of callback structure
def example(param="value"):
  """
  This is just a model of callback.
  """
  def example_cb(state, logger, model, optimizer):
    return state

  return example_cb

# The real callbacks

def print_scores_on_epoch():
  """
  write the scores in the log file
  """
  def print_scores_cb(state, logger, model, optimizer):
    e = state["i"] + 1
    ner = state["history"]["ner"][-1]
    f_score = state["history"]["f_score"][-1]
    precision_score = state["history"]["precision"][-1]

    logger.info("......................................................................")
    logger.info(f" Epoch N° {e}")
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


def reduce_lr_on_plateau(step=0.001, epochs=4, diff=1, score="f_score"):
  """
  Whe the model is not getting better scores (plateau or decrease)
  from the selected amount of last epochs this function sets the
  learning rate decreasing it a fixed step.
  Note: Uses the average of last scores
  :param step fixed amout to decrease the learning rate
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  """
  def reduce_lr_on_plateau_cb(state, logger, model, optimizer):
    if len(state["history"][score]) > epochs and state["lr"] > step:
      delta = np.diff(state["history"][score])[-epochs:]
      print(delta, " - promedio: ", np.mean(delta))
      if np.mean(delta) < diff:
        state["lr"] -= step
        logger.info(f"Not learning then reduce learn rate to {state['lr']}")

    return state

  return reduce_lr_on_plateau_cb


def early_stop(epochs=10, score="f_score", diff=5):
  """
  Sets the stop value to True in state if score is not improving during the last epochs
  Note: Uses the average of last scores
  :param epochs last epochs to be considered
  :param diff score difference amount to produce a change in the the learning rate
  :param score score used to calculate the diff
  """
  def early_stop_cb(state, logger, model, optimizer):
    if len(state["history"][score]) > epochs:
      delta = np.diff(state["history"][score])[-epochs:]
      if np.mean(delta) < diff:
        state["stop"] = True
        logger.info(f"Not learning what I want 😭😭. Bye Bye Adieu!")

    return state

  return early_stop_cb