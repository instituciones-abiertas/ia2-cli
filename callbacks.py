def print_scores_on_epoch():
  def print_scores_cb(state, logger, model, optimizer):
    e = state["i"] + 1
    ner = state["history"]["ner"][-1]
    f_score = state["history"]["f_score"][-1]
    precision_score = state["history"]["precision"][-1]

    logger.info("......................................................................")
    logger.info(f" Epoch N° {e}")
    logger.info(f"Losses rate: ner:{ner}, f1-score: {f_score}, precision: {precision_score}")

  return print_scores_cb


def save_best_model(path_best_model="", threshold=70, score="f_score"):
  def save_best_model_cb(state, logger, model, optimizer):
    # print("last f1: ", state["history"][score][-1], " | max: ", state["max_"+score])
    
    if (state["history"][score][-1] > threshold)  and  (state["history"][score][-1] > state["max_"+score]):
        e = state["i"]+1
        with model.use_params(optimizer.averages):
            model.to_disk(path_best_model)
            logger.info(f"💾 Saving model for epoch {e}")
     

  return save_best_model_cb


def reduce_lr_on_plateau(step=0.001, epochs=3, diff=1):
  def reduce_lr_on_plateau_cb(state, logger, model, optimizer):
    pass

  return reduce_lr_on_plateau_cb


def early_stop(epochs=5):
  def early_stop_cb(state, logger, model, optimizer):
    pass

  return early_stop_cb


