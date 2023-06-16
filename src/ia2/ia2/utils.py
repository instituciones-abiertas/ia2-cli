import shutil
import logging
import datetime
import subprocess

logger = logging.getLogger("Spacy cli utils.py")


def set_optimizer(optimizer, learn_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, L2=1e-3, max_grad_norm=1.0):
    """
    Function to customizer spaCy default Adam optimizer
    # read this before touch here https://enrico-alemani.medium.com/the-customized-spacy-training-loop-9e3756fbb6f6
    """

    optimizer.learn_rate = learn_rate
    optimizer.beta1 = beta1
    optimizer.beta2 = beta2
    optimizer.eps = eps
    optimizer.L2 = L2
    optimizer.max_grad_norm = max_grad_norm

    return optimizer


def set_dropout(train_config, FUNC_MAP):
    # FIXME NOT working with decaying
    if "dropout" not in train_config:
        return 0.2
    else:
        if type(train_config["dropout"]) == int or type(train_config["dropout"]) == float:
            return train_config["dropout"]
        else:
            d = train_config["dropout"]
            return FUNC_MAP[d.pop("f")](d["from"], d["to"], d["rate"])


def set_batch_size(train_config, FUNC_MAP):
    if "batch_size" not in train_config:
        return 4, ()
    else:
        if type(train_config["batch_size"]) == int or type(train_config["batch_size"]) == float:
            return train_config["batch_size"], ()
        else:
            b = train_config["batch_size"]
            return (FUNC_MAP[b.pop("f")], (b["from"], b["to"], b["rate"]))


def save_state_history(
    state,
    numero_losses,
    f_score,
    recall_score,
    precision_score,
    per_type_score,
    val_f_score,
    val_recall_score,
    val_precision_score,
    val_per_type_score,
    learn_rate,
    num_batches,
    dropout,
):
    state["history"]["ner"].append(numero_losses)
    state["history"]["f_score"].append(f_score)
    state["history"]["recall"].append(recall_score)
    state["history"]["precision"].append(precision_score)
    state["history"]["per_type_score"].append(per_type_score)

    # validation
    state["history"]["val_f_score"].append(val_f_score)
    state["history"]["val_recall"].append(val_recall_score)
    state["history"]["val_precision"].append(val_precision_score)
    state["history"]["val_per_type_score"].append(val_per_type_score)

    state["history"]["lr"].append(learn_rate)
    state["history"]["batches"].append(num_batches)
    state["history"]["dropout"].append(dropout)

    # this works if  iteration callacks are called later
    state["history"]["saved"].append("")


def run_command_with_timer(*args):
    """
    Calculate the time spend to run command

    :param *args: run command
    """
    begin_time = datetime.datetime.now()
    logger.info("####START####")
    logger.info(f"Start process {begin_time} ")
    subprocess.call(args[0], shell=True)
    end = datetime.datetime.now()
    logger.info(f"End {end} to process ")
    logger.info(f"Spend {end-begin_time} to process ")
