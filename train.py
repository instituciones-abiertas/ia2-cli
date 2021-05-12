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
import shutil
import utils
import functools
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

from spacy.pipeline import EntityRuler
from pipeline_components.entity_ruler import fetch_ruler_patterns_by_tag
from pipeline_components.entity_matcher import (
    ArticlesMatcher,
    EntityMatcher,
    ViolenceContextMatcher,
    matcher_patterns,
    fetch_cb_by_tag,
)
from pipeline_components.entity_custom import EntityCustom

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

    def calculate_by_entity(self, totalizer, entities):
        for span in entities:
            try:
                totalizer[span[2]] = totalizer[span[2]] + 1
            except:
                totalizer[span[2]] = 1


    def remove_misaligned_annotations_from(self, model_path: str, input_files_path: str, data_type: str):
        """
        Given a Spacy model path, a data file path and a data type (it should be one of: training, validation, testing), 
        removes misaligned annotations to the given file and updates the model to the given path.
        Before removing misaligned annotations, it creates a copy of the involved file.

        :param model_path: A model path
        :param input_files_path: A input file path to be "depured"
        :param data_type: file data type (it should be one of: training, validation, testing)
        """

        #we create a backup copy of the file to be modified
        dest = input_files_path.replace('.json', f'_copy.json')
        orig = input_files_path
        shutil.copyfile(orig, dest)

        nlp = spacy.load(model_path)

        misaligned_docs_qty = 0
        misaligned_lost_by_entities = {}
        only_misaligneds = 0
        total_lost_misaligned = 1

        with open(input_files_path, 'r') as f:
            data = json.load(f)

            for text, annotations in data:
                doc_gold_text = nlp.make_doc(text)
                annotation_ents = annotations.get('entities')
                alignment_values = spacy.gold.biluo_tags_from_offsets(doc_gold_text, annotation_ents)
                is_misaligned_doc = True if '-' in alignment_values else False
                if is_misaligned_doc:
                    text_array = nlp(text)
                    misaligned_texts = self.get_misaligned_texts(alignment_values, text_array)
                    for i, annot in enumerate(annotation_ents):
                        annotation_text = str(text[annot[0]:annot[1]])
                        if any(annotation_text.replace(' ', '') in text for text in misaligned_texts):
                            # print(f"annotation_text {annotation_text}")
                            # print(f"esta MAL la annotation: {annot} con idx: {i}")
                            annotation_ents.pop(i)
                            only_misaligneds = only_misaligneds + 1

                    misaligned_docs_qty = misaligned_docs_qty + 1
                    self.calculate_by_entity(misaligned_lost_by_entities, annotation_ents)

            with open(input_files_path, 'w') as f:
                # import pdb; pdb.set_trace()
                json.dump(data, f)

            if misaligned_docs_qty:
                total_lost_misaligned = functools.reduce(lambda a,b: a+b, misaligned_lost_by_entities.values())

            print(f'\n\nMisaligned docs for {data_type} data: {misaligned_docs_qty}/{len(data)} ({round(100*misaligned_docs_qty/len(data),2)}%).')
            print(f'Entities that could be lost because of misaligned: {total_lost_misaligned}.')
            print(f'Misaligned annotations removed: {only_misaligneds} ({round(only_misaligneds/total_lost_misaligned*100, 2)}% of total entities in related docs).')
            print(f'Data without misaligned annotations saved in:  {input_files_path}')


    # =================================
    # Model Training functions
    # =================================

    def get_best_model(
        self,
        optimizer,
        nlp,
        n_iter,
        training_data,
        path_best_model,
        save_misaligneds_to_file,
        validation_data=[],
        testing_data=[],
        callbacks={},
        settings={},
        disabled_pipes=[],
    ):
        init_time = time.time()
        print("\nsettings", settings)
        optimizer = utils.set_optimizer(optimizer, **settings["optimizer"])
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
                "saved": [],
            },
            "min_ner": 0,
            "max_f_score": 0,
            "max_recall": 0,
            "max_precision": 0,
            "max_val_f_score": 0,
            "max_val_recall": 0,
            "max_val_precision": 0,
            "lr": optimizer.learn_rate,
            "dropout": settings["dropout"],
            "elapsed_time": 0,
            "stop": False,
            "evaluate_test": False,
        }

        # for validation. activate if exists validation data
        if settings["evaluate"] == "val":
            val_texts, val_annotations = zip(*validation_data)

        if len(testing_data) > 0:
            test_texts, test_annotations = zip(*testing_data)

        tr_texts, tr_annotations = zip(*training_data)

        while not state["stop"] and state["i"] < state["epochs"]:
            # Randomizes training data
            random.shuffle(training_data)
            losses = {}

            # set/update Adam optimizer from state
            optimizer.learn_rate = state["lr"]

            if len(settings["batch_args"]) > 0:
                batch_size = settings["batch_size"](*settings["batch_args"])
            else:
                batch_size = settings["batch_size"]

            # Creates mini batches
            batches = minibatch(training_data, size=batch_size)
            num_batches = 0
            # bz = []
            for batch in batches:
                num_batches += 1
                texts, annotations = zip(*batch)

                # bz.append(len(texts))
                nlp.update(
                    texts,  # batch of raw texts
                    annotations,  # batch of annotations
                    drop=state["dropout"],
                    losses=losses,
                    sgd=optimizer,
                )

                # run batch callbacks
                for cb in callbacks["on_batch"]:
                    state = cb(state, logger, nlp, optimizer, disabled_pipes)
            log_annotations = True if state["i"] == 0 else False
            try:
                # compute validation scores
                val_f_score, val_precision_score, val_recall_score, val_per_type_score = -1, -1, -1, -1
                if settings["evaluate"] == "val":
                    logger.info("Evaluating docs from validation data")
                    val_f_score, val_precision_score, val_recall_score, val_per_type_score = self.evaluate_multiple(
                        optimizer, nlp, val_texts, val_annotations, "validation", save_misaligneds_to_file, log_annotations
                    )

                # train data score
                logger.info("Evaluating docs from training data")
                f_score, precision_score, recall_score, per_type_score = self.evaluate_multiple(
                    optimizer, nlp, tr_texts, tr_annotations, "training", save_misaligneds_to_file, log_annotations
                )

                numero_losses = losses.get("ner")

            except Exception as e:
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
                state = cb(state, logger, nlp, optimizer, disabled_pipes)

            # compute testing dataset scores
            # Since we can save many models if some threshold has been reached
            # during the train loop. We also want to get scores on test data
            # for each one of this models.
            if settings["evaluate"] == "test" and state["evaluate_test"]:
                logger.info("Evaluating docs from testing data")
                test_f_score, test_precision_score, test_recall_score, test_per_type_score = self.evaluate_multiple(
                    optimizer, nlp, test_texts, test_annotations, "test", save_misaligneds_to_file, True
                )
                logger.info("############################################################")
                logger.info("Evaluating saved model with test data")
                logger.info(f"Scores :f1-score: {test_f_score}, precision: {test_precision_score}")
                logger.info(f"{test_per_type_score}")
                logger.info("############################################################")

            state["evaluate_test"] = False
            state["i"] += 1

        # Run callbacks after train loop
        state["elapsed_time"] = (time.time() - init_time) / 60
        for cb in callbacks["on_stop"]:
            state = cb(state, logger, nlp, optimizer, disabled_pipes)

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
            # GPU
            # if GPU enabled

            if "use_gpu" in train_config and train_config["use_gpu"] is True:
                try:
                    import cupy

                    spacy.prefer_gpu()
                    logger.info("Using GPU 🎮")
                except Exception as e:
                    logger.warning("❗ Either GPU not available or CUDA versions is not compatible: %s", str(e))

            # test dataset
            evaluate = "val"
            if "evaluate" in train_config and train_config["evaluate"] == "test":
                evaluate = "test"

            test_ds = ""
            if "path_data_testing" in train_config:
                test_ds = train_config["path_data_testing"]

            # train settings
            dropout = utils.set_dropout(train_config, FUNC_MAP)
            batch_size, batch_args = utils.set_batch_size(train_config, FUNC_MAP)

            # optimizer hyperparams
            optimizer = {}
            if "optimizer" in train_config:
                optimizer = train_config["optimizer"]

            s = {
                "dropout": dropout,
                "optimizer": optimizer,
                # "dropout_args": dropout_args,
                "batch_size": batch_size,
                "batch_args": batch_args,
                "evaluate": evaluate,
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
            "",  # prefered save_best_model callback
            0,  # prefered save_best_model callback
            train_config["is_raw"],
            train_config["path_data_validation"],
            train_config["save_misaligneds_to_file"],
            path_data_testing=test_ds,
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
        save_misaligneds_to_file: bool = False,
        path_data_testing: str = "",
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
                logger.info(f"loading and converting validation data from dataturks: {path_data_training}")
                validation_data = convert_dataturks_to_spacy(path_data_training, ents)

            if path_data_testing != "":
                logger.info(f"loading and converting testing data from dataturks: {path_data_training}")
                testing_data = convert_dataturks_to_spacy(path_data_testing, ents)
        else:
            logger.info(f"loading pre-converted training data JSON: {path_data_training}")
            with open(path_data_training) as f:
                training_data = json.load(f)

            if path_data_validation != "":
                logger.info(f"loading pre-converted validation data JSON: {path_data_validation}")
                with open(path_data_validation) as f:
                    validation_data = json.load(f)

            if path_data_testing != "":
                logger.info(f"loading pre-converted testing data JSON: {path_data_testing}")
                with open(path_data_testing) as f:
                    testing_data = json.load(f)
                # mix train and validation data
                training_data += validation_data
            else:
                testing_data = []

        # print("total data: ", len(training_data))

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
        # Save disable pipelines for further restoring
        disabled_pipes = nlp.disable_pipes(*other_pipes)

        with warnings.catch_warnings():
            # Show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module="spacy")

            # NOTE these configs are not yet well documented in SpaCy 2
            # please read this https://github.com/explosion/spaCy/issues/5513#issuecomment-635169316
            optimizer = nlp.begin_training(component_cfg={"ner": {"conv_window": 3, "hidden_width": 64}})

            # this initial save remove the pipelines from base model and is not restoring them
            #
            # we save the first model and then we update it if there is a better version of it
            # logger.info(f"💾 Saving initial model")
            # nlp.to_disk(model_path)

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
                save_misaligneds_to_file,
                validation_data=validation_data,
                testing_data=testing_data,
                callbacks=callbacks,
                settings=settings,
                disabled_pipes=disabled_pipes,
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
            #FIXME  when training is throwing an error
            alignment_values = spacy.gold.biluo_tags_from_offsets(doc_gold_text, entity_ocurrences.get("entities"))
            is_misaligned_doc = True if '-' in alignment_values else False
            gold = GoldParse(doc_gold_text, entities=entity_ocurrences.get("entities"))
            pred_value = nlp(text)
            scorer.score(pred_value, gold)
            return scorer.scores, is_misaligned_doc, alignment_values
        except Exception as e:
            print(e)

    def get_misaligned_texts(self, misaligned_array, text_array):
        """
        Given a list of alignment values for a doc and an array with the tokenized doc, 
        it find and returns an array of matching misaligned annotations.

        :param misaligned_array: A list of alignment values for a doc.
        :param text_array: A list that represents a tokenized doc.
        """        
        misaligned_indexes = [i for i, x in enumerate(misaligned_array) if x == "-"]
        misaligned_texts = []
        i = 0
        word_ended = False
        next_index_is_consecutive = False
        text = ""

        while i < len(misaligned_indexes):
            token_text = text_array[misaligned_indexes[i]].text
            try:
                next_index_is_consecutive = len(misaligned_indexes) > 1 and (misaligned_indexes[i+1] - misaligned_indexes[i]) == 1
            except:
                next_index_is_consecutive = False

            if i == 0 or text == "":
                it_follows_prev_index = False 
                text = token_text
            else:                
                it_follows_prev_index = (misaligned_indexes[i] - misaligned_indexes[i-1]) == 1
            
            if it_follows_prev_index:
                text = text + " "+ token_text

            word_ended = not next_index_is_consecutive

            if word_ended:
                misaligned_texts.append(text.replace(' ', ''))
                text = ""
            
            i = i+1

        return misaligned_texts

    def get_total_misaligneds(self, nlp, save_it_to_file = False, filename="misaligned.json", misaligned_docs=[]):
        """
        IMPORTANT! This function is being called when training the model.

        Given a filename and an array of misaligned docs, it finds for every doc only 
        matching misaligned annotations and saves them to a json file with a structure similar
        to the one used by Spacy for training data (json object with text and annotations).
        The file/s are being saved in logs folder.

        :param filename: A filename for the json file that is going to be saved.
        :param misaligned: A list of misaligned docs.
        """
        path = f"logs/{filename}"
        logger.info("\n\n")
        if save_it_to_file:
            logger.info(f"[get_total_misaligneds] 💾 will be saved in a {path} file")
            # create file if not exists
            if not os.path.exists(path):
                with open(path, "w"):
                    pass

        data = []
        total_misaligneds = 0
        for i in range(len(misaligned_docs)):
            text_raw = misaligned_docs[i]["text"]
            text_array = nlp(text_raw)
            #to check how the doc is being tokenized and understand why the misaligned warning is arising
            # if "validation" in filename: 
                # tok_exp = nlp.tokenizer.explain(text_raw)
                # for t in tok_exp:
                    # print(t[1], "\t", t[0])
            misaligned_texts = self.get_misaligned_texts(misaligned_docs[i]["alignment_values"], text_array)
            total_misaligneds = total_misaligneds + len(misaligned_texts)

            if save_it_to_file:
                annotations = misaligned_docs[i]["entities"]["entities"]
                misaligned_annotations = []

                for i, annot in enumerate(annotations):
                    annotation_text = str(text_raw[annot[0]:annot[1]])
                    if any(annotation_text.replace(' ', '') in text for text in misaligned_texts):
                        annotation = annotations[i]
                        annotation.append(annotation_text)
                        misaligned_annotations.append(annotation)
                #we save a part of the text in order to be able to search it in the full text file
                data.append({"text": text_raw[0:700], "annotations": misaligned_annotations})

        if save_it_to_file:
            srsly.write_json(path, data)
            logger.info(f"\n[] {path} file saved!")

        return total_misaligneds


    def evaluate_multiple(self, optimizer, nlp, texts: list, entity_occurences: list, data_type: str, save_misaligneds_to_file: bool, log_annotations: bool):
        f_score_sum = 0
        precision_score_sum = 0
        recall_score_sum = 0
        qty_misaligned_docs = 0
        ents_per_type_sum = {}
        misaligned_docs = []
        total_by_entities = {}
        misaligned_lost_by_entities = {}
        only_misaligneds = 0
        total_lost_misaligned = 1

        for idx in range(len(texts)):
            text = texts[idx]
            entities_for_text = entity_occurences[idx]
            with nlp.use_params(optimizer.averages):
                scores, is_misaligned_doc, alignment_values = self.evaluate(nlp, text, entities_for_text)
                recall_score_sum += scores.get("ents_r")
                precision_score_sum += scores.get("ents_p")
                f_score_sum += scores.get("ents_f")

                self.calculate_by_entity(total_by_entities, entities_for_text['entities'])

                if is_misaligned_doc:
                    misaligned_docs.append({"text": text, "entities":entities_for_text, "alignment_values": alignment_values})
                    self.calculate_by_entity(misaligned_lost_by_entities, entities_for_text['entities'])

                for key, value in scores["ents_per_type"].items():
                    if key not in ents_per_type_sum:
                        ents_per_type_sum[key] = value["f"]
                    else:
                        ents_per_type_sum[key] += value["f"]

        if len(misaligned_docs):
            only_misaligneds = self.get_total_misaligneds(nlp, save_misaligneds_to_file, f"{data_type}_misaligned_docs.json", misaligned_docs)
            total_lost_misaligned = functools.reduce(lambda a,b: a+b, misaligned_lost_by_entities.values())

        if log_annotations:
            logger.info(f'Misaligned docs for {data_type} data: {len(misaligned_docs)}/{len(texts)} ({round(100*len(misaligned_docs)/len(texts),2)}%).')
            logger.info(f'ANNOTATIONS!')
            logger.info(f'Total by entities: {total_by_entities}.')
            logger.info(f'Lost entities because of misaligned: {total_lost_misaligned}.')
            logger.info(f'Misaligned: {only_misaligneds} ({round(only_misaligneds/total_lost_misaligned*100, 2)}%).')

        for key, value in ents_per_type_sum.items():
            ents_per_type_sum[key] = value / len(texts)

        return (
            f_score_sum / len(texts),
            precision_score_sum / len(texts),
            recall_score_sum / len(texts),
            ents_per_type_sum
        )

    def build_model_package(
        self, model_path: str, package_path: str, model_name: str, model_version: str, model_components: str
    ):
        """
        Given a model path, a dist path, a model name, a version number and a
        directory to pipelines, adds those pipelines to the given model, assigns
        a name to it and build a python package in the dist directory.

        :param model_path: A model path
        :param package_path: A package path
        :param model_name: A new model name
        :param model_version: A new model version
        :param model_components: A model components path
        """
        nlp = spacy.load(model_path)

        nlp.meta["name"] = model_name
        nlp.meta["version"] = str(model_version)
        pipelines_tag = "todas"

        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns(fetch_ruler_patterns_by_tag(pipelines_tag))
        nlp.add_pipe(ruler)

        entity_matcher = EntityMatcher(
            nlp, matcher_patterns, after_callbacks=[cb(nlp) for cb in fetch_cb_by_tag(pipelines_tag)]
        )
        nlp.add_pipe(entity_matcher)

        entity_custom = EntityCustom(nlp, pipelines_tag)
        nlp.add_pipe(entity_custom)

        nlp.to_disk(model_path)
        logger.info(f'Succesfully added rule based mathching at model: "{model_path}".')

        package(model_path, package_path)
        logger.info(f'Succesfully package at model: "{package_path}".')

        package_name = nlp.meta["lang"] + "_" + nlp.meta["name"]
        package_dir = package_name + "-" + nlp.meta["version"]
        package_base_path = os.path.join(package_path, package_dir, package_name)

        # Copio archivos con modulos custom al directorio pipe_components (no uso model_components para que sea fijo y siempre el mismo en nuestro componente, y asi poder desacoplarlo de cuando tengamos multiples clientes
        package_components_dir = "pipeline_components"
        files_src = ["entity_matcher.py", "entity_custom.py"]
        dest_component_dir = os.path.join(package_base_path, package_dir, package_components_dir)
        os.mkdir(dest_component_dir)
        for f in files_src:
            dest = os.path.join(dest_component_dir, f)
            orig = os.path.join(model_components, f)
            shutil.copyfile(orig, dest)

        # NOTE indentation in this string leads to model error
        new_code = f"""import os
import importlib
from spacy.language import Language
def import_path(path):
    module_name = os.path.basename(path).replace('-', '_').replace('.','-')
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dir =  os.fspath(Path(__file__).parent)
moduloMatcher = import_path(dir + "/{package_dir}/{package_components_dir}/entity_matcher.py")
moduloCustom = import_path(dir + "/{package_dir}/{package_components_dir}/entity_custom.py")

Language.factories['entity_matcher'] = lambda nlp, **cfg: moduloMatcher.EntityMatcher(nlp, moduloMatcher.matcher_patterns, after_callbacks=[ cb(nlp) for cb in moduloMatcher.fetch_cb_by_tag("todas") ])
Language.factories['entity_custom'] = lambda nlp, **cfg: moduloCustom.EntityCustom(nlp, "todas")
"""

        insert_line = 6
        package_filename = "__init__.py"
        init_py = os.path.join(package_base_path, package_filename)
        with open(init_py, "r") as f:
            content = f.readlines()
        with open(init_py, "w") as f:
            for c in new_code.splitlines()[::-1]:
                content.insert(insert_line, c + "\n")
            f.writelines(content)
        logger.info("Succesfully write language factories to model")

    def show_text(self, files_path: str, entity: str, context_words: int = 0):
        """
        Given the path to a dataturks .json format input file directory and an
        entity name, prints the annotation text from label.

        :param files_path: Directory pointing to dataturks .json files
        :param entity: entity label name.
        :param context_words: integer for nbor words.
        """

        W = "\033[0m"  # white (normal)
        R = "\033[31m"  # red
        G = "\033[32m"  # green
        files = [os.path.join(files_path, f) for f in listdir(files_path) if isfile(join(files_path, f))]
        texts = []

        for file_ in files:
            with open(file_, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    for a in data["annotation"] or []:
                        output = ""
                        if a["label"][0] == entity:
                            if not a["points"][0]["text"] in texts:
                                text = a["points"][0]["text"]
                                if context_words:
                                    text = re.escape(a["points"][0]["text"])
                                    interval = r"{{0,{0}}}".format(context_words)
                                    regex = (
                                        r"((?:\S+\s+)"
                                        + interval
                                        + r"\b"
                                        + text
                                        + r"\b\s*(?:\S+\b\s*)"
                                        + interval
                                        + ")"
                                    )
                                    x = re.search(regex, data["content"])
                                    if x:
                                        output = x.group()
                                        output = output.replace(text.replace("\\", ""), R + text + W)
                                else:
                                    output = text
                                posicion = " -- Start: {}{} {}End: {}{}{}".format(
                                    G, str(a["points"][0]["start"]), W, G, str(a["points"][0]["end"]), W
                                )
                                texts.append(output.replace("\\", "") + posicion)

        for text in texts:
            print(text)

if __name__ == "__main__":
    fire.Fire(SpacyUtils)
