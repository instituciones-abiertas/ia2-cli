import fire
import warnings
import pickle
import logging
import json
import random
import datetime
import os
import re
import sys
import spacy
import time
import utils
from spacy.util import minibatch, compounding, decaying
from spacy.scorer import Scorer
from spacy.gold import GoldParse
from spacy.cli import package
import srsly
from os import listdir
from os.path import isfile, join
from callbacks import (
    print_scores_on_epoch,
    save_best_model,
    reduce_lr_on_plateau,
    early_stop,
    update_best_scores,
    sleep,
    log_best_scores,
    save_csv_history,
    change_dropout_fixed,
)


logger = logging.getLogger("Spacy cli util")
logger.setLevel(logging.DEBUG)
logger_fh = logging.FileHandler("logs/debug.log")
logger_fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] (%(name)s) :: %(levelname)s :: %(message)s")
logger_fh.setFormatter(formatter)
logger.addHandler(logger_fh)


def convert_dataturks_to_spacy(dataturks_JSON_file_path, entity_list):
    try:
        training_data = []
        lines = []

        with open(dataturks_JSON_file_path, "r") as f:
            lines = f.readlines()
            count_total = 0
            count_overlaped = 0
            for line in lines:
                data = json.loads(line)
                text = data["content"]
                entities = []
                annotations = []
                if data["annotation"]:
                    for annotation in data["annotation"]:
                        point = annotation["points"][0]
                        label = annotation["label"]
                        if label[0] in entity_list:
                            annotations.append(
                                (
                                    point["start"],
                                    point["end"],
                                    label,
                                    point["end"] - point["start"],
                                )
                            )
                    annotations = sorted(annotations, key=lambda student: student[3], reverse=True)

                    seen_tokens = set()
                    for annotation in annotations:

                        start = annotation[0]
                        end = annotation[1]
                        labels = annotation[2]
                        if start not in seen_tokens and end - 1 not in seen_tokens:
                            seen_tokens.update(range(start, end))
                            if isinstance(labels, list):
                                labels = labels[0]
                            entities.append((start, end + 1, labels))
                            count_total = count_total + 1
                        else:
                            count_overlaped = count_overlaped + 1
                            # logger.info("{} {} {} is overlapped".format(start, end, labels))
                training_data.append((text, {"entities": entities}))

        # logger.info("Entities: {}".format(count_total))
        # logger.info("Overlapped entities : {}".format(count_overlaped))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_file_path + "\n" + "error = " + str(e))
        return None

class SpacyUtils:
    """
    SpacyUtils: Dataturks format converter and other Spacy model utilities.

    Methods
    -------

    + create_blank_model(path_save_model: str)
        Creates a blank Spacy model.
    + create_custom_spacy_model(spacy_model: str, path_save_model: str)
        Creates a model that extends from: "es_core_news_sm", "es_core_news_md", "es_core_news_lg".
    + add_new_entity_to_model(ents: list, model_path: str)
        Adds entities to an existing model.
    + train_model(path_data_training: str, n_iter: int, model_path: str, ents: list)
        Given a training dataset, trains an existing Spacy model. Accepts:
        iterations number and list of entities.
    """

    # =================================
    # Creation Models functions
    # =================================

    def create_blank_model(self, output_path: str):
        """
        Given an output path creates a blank model using the "es" language.
        :param output_path: A string representing an output path.

        """
        nlp = spacy.blank("es")
        nlp.to_disk(output_path)
        logger.info(f'Succesfully created model at: "{output_path}..."')

    def create_custom_spacy_model(self, spacy_model: str, output_path: str):
        """
        Given a Spacy model name and an output path, loads the model using the
        Spacy api and saves it at the given path.

        :param spacy_model: A string representing a Spacy model. E.g.:
        "es_core_news_lg".
        :param output_path: A string representing the output path where the
        model should be written.
        """
        nlp = spacy.load(spacy_model)
        nlp.to_disk(output_path)
        logger.info(f'Succesfully created model at: "{output_path}".')

    # =================================
    # Update Models functions
    # =================================

    def add_new_entity_to_model(self, ents: list, model_path: str):
        """
        Given a list of entities and a Spacy model path, adds every entity to
        the model and updates the model to the given path.

        :param ents: A list of string representing entites
        :param model_path: A model path
        """
        arr = sys.argv[2].split(",")
        nlp = spacy.load(model_path)
        if "ner" not in nlp.pipe_names:
            component = nlp.create_pipe("ner")
            nlp.add_pipe(component, last=True)
        ner = nlp.get_pipe("ner")
        for ent in arr:
            ner.add_label(ent)
        print(f"Entities add to model {ner.move_names}")
        nlp.begin_training()
        nlp.to_disk(model_path)

        logger.info(f'Succesfully added entities at model: "{model_path}".')

    # =================================
    # Data Conversion functions
    # =================================

    def convert_dataturks_to_train_file(
        self, input_files_path: str, entities: list, output_file_path: str, num_files: int = 0
    ):
        """
        Given an input directory and a list of entities, converts every .json
        file from the directory into a single .json to be used with train_model
        method. Unlike utils.py - convert_dataturks_to_training_cli this does not convert
        documents to Spacy JSON format used in Spacy cli train command.
        Writes it out to the given output directory.

        :param input_files_path: Directory pointing to dataturks .json files to
        be converted.
        :param entities: A list of entities, separated by comma, to be
        considered during annotations extraction from each dadaturks batch.
        :param output_file_path: The path and name of the output file.
        :param num_files An integer which means the numbers of files from
        input path to be included. 0 means all files and is default option
        """

        TRAIN_DATA = []
        begin_time = datetime.datetime.now()
        input_files = [f for f in listdir(input_files_path) if isfile(join(input_files_path, f))]

        if num_files == 0:
            num_files = len(input_files)

        for input_file in input_files[:num_files]:
            logger.info(f'Extracting raw data and occurrences from file: "{input_file}"...')
            extracted_data = convert_dataturks_to_spacy(f"{input_files_path}/{input_file}", entities)
            TRAIN_DATA = TRAIN_DATA + extracted_data
            logger.info(f'Finished extracting data from file "{input_file}".')

        diff = datetime.datetime.now() - begin_time
        logger.info(f"Lasted {diff} to extract dataturks data from {len(input_files)} documents.")
        logger.info(
            f"Converting {len(TRAIN_DATA)} Documents with Occurences extracted from {len(input_files)} files into Spacy supported format..."
        )
        try:
            logger.info(f'💾 Writing final output at "{output_file_path}"...')
            srsly.write_json(output_file_path, TRAIN_DATA)
            logger.info("💾 Done.")
        except Exception:
            logging.exception(f'An error occured writing the output file at "{output_file_path}".')


    # =================================
    # Model Training functions
    # =================================

    def get_best_model(
        self, optimizer, nlp, n_iter, training_data, path_best_model, validation_data=[], callbacks={}, settings={}
    ):
        init_time = time.time()
        print("\nsettings", settings)

        state = {
            "i": 0,
            "epochs": n_iter,
            "train_size": len(training_data),
            "history": {
                "ner": [],
                "f_score": [],
                "recall": [],
                "precision": [],
                "per_type_score": [],
                "val_f_score": [],
                "val_recall": [],
                "val_precision": [],
                "val_per_type_score": [],
                "lr": [],
                "batches": [],  # processed batches
                "dropout": [],
            },
            "min_ner": 0,
            "max_f_score": 0,
            "max_recall": 0,
            "max_precision": 0,
            "max_val_f_score": 0,
            "max_val_recall": 0,
            "max_val_precision": 0,
            "lr": settings["lr"],
            "beta1": settings["beta1"],
            "dropout": settings["dropout"],
            "elapsed_time": 0,
            "stop": False,
        }


        optimizer.L2 = 0.0
        # callback
        # callbacks["on_iteration"].append(update_best_scores())

        # for validation
        val_texts, val_annotations = zip(*validation_data)
        tr_texts, tr_annotations = zip(*training_data)

        while not state["stop"] and state["i"] < state["epochs"]:
            # Randomizes training data
            random.shuffle(training_data)
            losses = {}

            # set/update Adam optimizer from state
            optimizer.learn_rate = state["lr"]
            optimizer.beta1 = state["beta1"]

            if len(settings["batch_args"]) > 0:
                batch_size = settings["batch_size"](*settings["batch_args"])
            else:
                batch_size = settings["batch_size"]

            # Creates mini batches
            batches = minibatch(training_data, size=batch_size)
            num_batches = 0
            bz = []
            for batch in batches:
                num_batches += 1
                texts, annotations = zip(*batch)

                bz.append(len(texts))
                nlp.update(
                    texts,  # batch of raw texts
                    annotations,  # batch of annotations
                    drop=state["dropout"],
                    losses=losses,
                    sgd=optimizer,
                )

                # run batch callbacks
                for cb in callbacks["on_batch"]:
                    state = cb(state, logger, nlp, optimizer)

            try:
                print(bz)
                # compute validation scores
                val_f_score, val_precision_score, val_recall_score, val_per_type_score = self.evaluate_multiple(
                    optimizer, nlp, val_texts, val_annotations
                )

                # train data score
                f_score, precision_score, recall_score, per_type_score = self.evaluate_multiple(
                    optimizer, nlp, tr_texts, tr_annotations
                )
                numero_losses = losses.get("ner")

            except Exception:
                logger.exception("The batch has no training data.")

            utils.save_state_history(
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
                optimizer.learn_rate,
                num_batches,
                state["dropout"],
            )

            # run callbacks after each iteration

            for cb in callbacks["on_iteration"]:
                state = cb(state, logger, nlp, optimizer)

            state["i"] += 1

        # Run callbacks after train loop
        state["elapsed_time"] = (time.time() - init_time) / 60
        for cb in callbacks["on_stop"]:
            state = cb(state, logger, nlp, optimizer)

    def train(self, config: str):
        """
        Runs the train_model method using the selected configuration from train_config.json
        :param config the train configuration name
        """
        FUNC_MAP = {
            "save_best_model": save_best_model,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "early_stop": early_stop,
            "update_best_scores": update_best_scores,
            "log_best_scores": log_best_scores,
            "save_csv_history": save_csv_history,
            "sleep": sleep,
            "print_scores_on_epoch": print_scores_on_epoch,
            "change_dropout_fixed": change_dropout_fixed,
            # spacy funcs
            "compounding": compounding,
            "decaying": decaying,
        }
        try:
            with open("train_config.json") as f:
                train_config = json.load(f)
                train_config = train_config[config]

            # train settings
            lr, beta1 = utils.set_lr_and_beta1(train_config)
            dropout = utils.set_dropout(train_config, FUNC_MAP)
            batch_size, batch_args = utils.set_batch_size(train_config, FUNC_MAP)

            s = {
                "lr": lr,
                "beta1": beta1,
                "dropout": dropout,
                # "dropout_args": dropout_args,
                "batch_size": batch_size,
                "batch_args": batch_args,
            }

            on_iter_cb = []
            for cb in train_config["callbacks"]["on_iteration"]:
                on_iter_cb.append(FUNC_MAP[cb.pop("f")](**cb))

            on_batch_cb = []
            for cb in train_config["callbacks"]["on_batch"]:
                on_batch_cb.append(FUNC_MAP[cb.pop("f")](**cb))

            on_stop_cb = []
            for cb in train_config["callbacks"]["on_stop"]:
                on_stop_cb.append(FUNC_MAP[cb.pop("f")](**cb))

            c = {"on_iteration": on_iter_cb, "on_batch": on_batch_cb, "on_stop": on_stop_cb}

        except Exception as e:
            print(e)

        print(f"Training with {config} configuration")

        return self.train_model(
            train_config["path_data_training"],
            train_config["epochs"],
            train_config["model_path"],
            train_config["entities"],
            train_config["save_model_path"],
            train_config["threshold"],
            train_config["is_raw"],
            train_config["path_data_validation"],
            callbacks=c,
            settings=s,
            train_subset=train_config["train_subset"],
        )

    def train_model(
        self,
        path_data_training: str,
        n_iter: int,
        model_path: str,
        ents: list,
        path_best_model: str,
        max_losses: float,
        is_raw: bool = True,
        path_data_validation: str = "",
        callbacks={},
        settings={},
        train_subset=0,
    ):
        """
        Given a dataturks .json format input file, a list of entities and a path
        to an existent Spacy model, trains that model for `n_iter` iterations
        with the given entities. Whenever a best model is found it is writen to
        disk at the given output path.

        :param path_data_training: A string representing the path to the input
        file.
        :param n_iter: An integer representing a number of iterations.
        :param model_path: A string representing the path to the model to train.
        :param ents: A list of string representing the entities to consider
        during training.
        :param path_best_model: A string representing the path to write the best
        trained model.
        :param max_losses: A float representing the maximum NER losses value
        to consider before start writing best models output.
        :param is_raw A boolean that determines if the train file will be converted        True by default
        """

        if is_raw:
            logger.info(f"loading and converting training data from dataturks: {path_data_training}")
            training_data = convert_dataturks_to_spacy(path_data_training, ents)
            if path_data_validation != "":
                validation_data = convert_dataturks_to_spacy(path_data_training, ents)
        else:
            logger.info(f"loading pre-converted training data JSON: {path_data_training}")
            with open(path_data_training) as f:
                training_data = json.load(f)

            if path_data_validation != "":
                logger.info(f"loading pre-converted validation data JSON: {path_data_validation}")
                with open(path_data_validation) as f:
                    validation_data = json.load(f)

        nlp = spacy.load(model_path)
        # Filters pipes to disable them during training
        pipe_exceptions = ["ner"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        # Creates a ner pipe if it does not exist.
        if "ner" not in nlp.pipe_names:
            component = nlp.create_pipe("ner")
            nlp.add_pipe(component)
        ner = nlp.get_pipe("ner")

        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
        # Save disable pipelines
        disabled_pipes = nlp.disable_pipes(*other_pipes)

        with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
            # Show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module="spacy")
            optimizer = nlp.begin_training()

            # we save the first model and then we update it if there is a better version of it
            logger.info(f"💾 Saving initial model")
            nlp.to_disk(model_path)

            # adding plugins for each step of train loop train loop
            # these are default callbacks
            if not bool(callbacks):
                callbacks = {
                    "on_batch": [sleep(secs=1)],
                    "on_iteration": [
                        print_scores_on_epoch(),
                        save_best_model(path_best_model=path_best_model, threshold=max_losses),
                        reduce_lr_on_plateau(epochs=3, diff=1, step=0.001),
                        early_stop(epochs=10, diff=2),
                        update_best_scores(),
                        sleep(secs=3, log=True),
                    ],
                    "on_stop": [log_best_scores(), save_csv_history()],
                }

            # TODO default settings just in case not using train_config.json
            if train_subset > 0:
                training_data = random.sample(training_data, train_subset)
                logger.info(f"Using a random subset of {train_subset} texts")


            self.get_best_model(
                optimizer,
                nlp,
                n_iter,
                training_data,
                path_best_model,
                validation_data=validation_data,
                callbacks=callbacks,
                settings=settings,
            )


    # =================================
    # Evaluation and display functions
    # =================================

    def display_text_prediction(self, model_path: str, text: str):
        """
        Given a path to an existent Spacy model and a raw text, uses the model
        to output predictions and serve them at port 5030 using DisplayCy.

        :param model_path: A string representing the path to a Spacy model.
        :param text: A string representing raw text for the model to predict
        results.
        """
        nlp = spacy.load(model_path, disable=["tagger", "parser"])
        doc = nlp(text)
        spacy.displacy.serve(doc, style="ent", page=True, port=5030)

    def evaluate(self, nlp, text: str, entity_ocurrences: list):
        """
        Given a path to an existent Spacy model, a raw text and a list of
        entity occurences, computes a Spacy model score to return the Scorer
        scores.

        :param model_path: A string representing the directory of an existent
        Spacy model.
        :param text: A raw text to use as evaluation data.
        :param entity_occurences: A list of entity occurrences.
        """
        scorer = Scorer()
        try:
            doc_gold_text = nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=entity_ocurrences.get("entities"))
            pred_value = nlp(text)
            scorer.score(pred_value, gold)
            return scorer.scores
        except Exception as e:
            print(e)

    def evaluate_multiple(self, optimizer, nlp, texts: list, entity_occurences: list):
        f_score_sum = 0
        precision_score_sum = 0
        recall_score_sum = 0
        per_type_sum = 0
        ents_per_type_sum = {}
        for idx in range(len(texts)):
            text = texts[idx]
            entities_for_text = entity_occurences[idx]
            with nlp.use_params(optimizer.averages):
                scores = self.evaluate(nlp, text, entities_for_text)
                recall_score_sum += scores.get("ents_r")
                precision_score_sum += scores.get("ents_p")
                f_score_sum += scores.get("ents_f")

                for key, value in scores["ents_per_type"].items():
                    if key not in ents_per_type_sum:
                        ents_per_type_sum[key] = value["f"]
                    else:
                        ents_per_type_sum[key] += value["f"]

        for key, value in ents_per_type_sum.items():
            ents_per_type_sum[key] = value / len(texts)

        return (
            f_score_sum / len(texts),
            precision_score_sum / len(texts),
            recall_score_sum / len(texts),
            ents_per_type_sum,
        )


if __name__ == "__main__":
    fire.Fire(SpacyUtils)
