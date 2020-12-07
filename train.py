import fire
import warnings
import pickle
import logging
import json
import random
import datetime
import os
import re
import spacy
from spacy.util import minibatch, compounding, filter_spans
from spacy.gold import biluo_tags_from_offsets
from spacy.scorer import Scorer
from spacy.gold import GoldParse, docs_to_json
import srsly
from os import listdir
from os.path import isfile, join

logger = logging.getLogger('Spacy cli util')
logger.setLevel(logging.DEBUG)
logger_fh = logging.FileHandler('logs/debug.log')
logger_fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] (%(name)s) :: %(levelname)s :: %(message)s')
logger_fh.setFormatter(formatter)
logger.addHandler(logger_fh)

# Sets a global default value for DROPOUT_RATE
DROPOUT_RATE = 0.2

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
                    annotations = sorted(
                        annotations, key=lambda student: student[3], reverse=True
                    )

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
        logging.exception(
            "Unable to process " + dataturks_JSON_file_path + "\n" + "error = " + str(e)
        )
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

    def create_blank_model(self, path_save_model: str):
        """
        Given an output path creates a blank model using the "es" language.
        :param path_save_model: A string representing an output path.  

        """
        nlp = spacy.blank("es")
        nlp.to_disk(path_save_model)
        logger.info("Succesfully created model at: \"{}...\"".format(
            path_save_model
        ))

    def create_custom_spacy_model(self, spacy_model: str, path_save_model: str):
        """
        Given a Spacy model name and an output path, loads the model using the
        Spacy api and saves it at the given path.

        :param spacy_model: A string representing a Spacy model. E.g.:
        "es_core_news_lg".  
        :param path_save_model: A string representing the output path where the
        model should be written.  
        """
        nlp = spacy.load(spacy_model)
        nlp.to_disk(path_save_model)
        logger.info("Succesfully created model at: \"{}\".".format(
            path_save_model
        ))

    def add_new_entity_to_model(self, ents: list, model_path: str):
        """
        Given a list of entities and a Spacy model path, adds every entity to
        the model and updates the model to the given path.

        :param ents: A list of string representing entites  
        :param model_path: A model path  
        """

        nlp = spacy.load(model_path)
        if "ner" not in nlp.pipe_names:
            component = nlp.create_pipe("ner")
            nlp.add_pipe(component)
        ner = nlp.get_pipe("ner")
        for ent in ents:
            ner.add_label(ent)
        print(ner.move_names)
        nlp.to_disk(model_path)

        logger.info("Succesfully added entities at model: \"{}\".".format(
            model_path
        ))

    def convert_dataturks_to_spacy(
        self, input_file_path: str, output_file_path: str, entities: list
    ):
        """
        Given a dataturks format .json file, an output path and a list of
        entities, converts the input data into Spacy recognisable format to
        pickle dump it at the given output path.

        :param input_file_path: A string representing the path to a dataturks
        .json format input file.  
        :param output_file_path: A string representing the output path.  
        :param entities: A list of string representing entities to recognise
        during data conversion.  
        """
        logger.info(f"Starts converting data from \"{input_file_path}\"...")
        training_data = []
        log = convert_dataturks_to_spacy(input_file_path, entities)
        with open(output_file_path, "a+") as f:
            training_data.append(convert_dataturks_to_spacy(input_file_path, entities))
        with open(output_file_path, "wb") as output:
            pickle.dump(training_data, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Succesfully converted data at \"{}\".".format(
            output_file_path
        ))

    def convert_dataturks_to_training_cli(self,
        input_files_path: str,
        entities: list,
        output_file_path: str
    ):
        """
        Given an input directory and a list of entities, converts every .json
        file from the directory into a single .json recognisable by Spacy and
        writes it out to the given output directory.

        :param input_files_path: Directory pointing to dataturks .json files to
        be converted.  
        :param entities: A list of entities, separated by comma, to be
        considered during annotations extraction from each dadaturks batch.  
        :param output_file_path: The path and name of the output file.  
        """
        nlp = spacy.load("es_core_news_lg", disable=["ner"])

        TRAIN_DATA = []
        begin_time = datetime.datetime.now()
        input_files = [
            f
            for f in listdir(input_files_path)
            if isfile(join(input_files_path, f))
        ]

        for input_file in input_files:
            logger.info("Extracting raw data and occurrences from file: \"{}\"...".format(input_file))
            extracted_data = convert_dataturks_to_spacy(
                f"{input_files_path}/{input_file}",
                entities
            )
            TRAIN_DATA = TRAIN_DATA + extracted_data
            logger.info("Finished extracting data from file \"{}\".".format(input_file))

        diff = datetime.datetime.now() - begin_time
        logger.info("Lasted {} to extract dataturks data from {} documents.".format(diff, len(input_files)))
        logger.info("Converting {} Documents with Occurences extracted from {} files into Spacy supported format...".format(len(TRAIN_DATA), len(input_files)))

        docs = []
        for text, annot in TRAIN_DATA:
            doc = nlp(text)

            new_ents = []
            for start_idx, end_idx, label in annot["entities"]:
                span = doc.char_span(start_idx, end_idx, label=label)

                if span is None:
                    conflicted_entity = {
                        "file_dir": input_files_path,
                        "label": label,
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "matches_text": text[start_idx:end_idx]
                    }
                    logger.critical(f"Conflicted entity: could not save an entity because it does not match an entity in the given document. Output: \"{conflicted_entity}\".")
                else:
                    new_ents.append(span)

            doc.ents = new_ents
            docs.append(doc)

        diff = datetime.datetime.now() - begin_time
        logger.info("Finished Converting {} Spacy Documents into trainable data. Lasted: ".format(len(TRAIN_DATA), diff))

        try:
            logger.info("💾 Writing final output at \"{}\"...".format(output_file_path))
            srsly.write_json(output_file_path, [docs_to_json(docs)])
            logger.info("💾 Done.")
        except Exception:
            logging.exception("An error occured writing the output file at \"{}\".".format(output_file_path))

    def train_model(
        self,
        path_data_training: str,
        n_iter: int,
        model_path: str,
        ents: list,
        path_best_model: str,
        max_losses: float,
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
        """
        best = max_losses
        training_data = convert_dataturks_to_spacy(path_data_training, ents)
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

        with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
            # Show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module="spacy")
            nlp.begin_training()

            for itn in range(n_iter):
                # Randomizes training data
                random.shuffle(training_data)
                losses = {}
                # Creates mini batches
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))

                for batch in batches:
                    texts, annotations = zip(*batch)

                    nlp.update(
                        # batch of raw texts
                        texts,
                        # batch of annotations
                        annotations,
                        drop=DROPOUT_RATE,
                        losses=losses,
                    )
                logger.info("⬇️ Losses rate: [{}]".format(losses))
                try:
                    numero_losses = losses.get("ner")
                    if numero_losses < best and numero_losses > 0:
                        best = numero_losses
                        nlp.to_disk(path_best_model)
                        logger.info("💾 Saving model with losses: [{}]".format(best))

                except Exception:
                    logger.exception("The batch has no training data.")

            nlp.to_disk(model_path)
            return best

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

    def evaluate(self, model_path: str, text: str, entity_occurences: list):
        """
        Given a path to an existent Spacy model, a raw text and a list of
        entity occurences, computes a Spacy model score to return the Scorer
        scores.
        """
        scorer = Scorer()
        nlp = spacy.load(model_path)
        try:
            doc_gold_text = nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=entity_occurences)
            pred_value = nlp(text)
            scorer.score(pred_value, gold)
            return scorer.scores
        except Exception as e:
            print(e)

    def train_all_files_in_folder(
        self,
        training_files_path: str,
        n_iter: int,
        model_path: str,
        entities: list,
        best_model_path: str,
        max_losses: float,
    ):
        """
        Given the path to a dataturks .json format input file directory, a list
        of entities and a path to an existent Spacy model, trains that model for
        `n_iter` iterations with the given entities. Whenever a best model is
        found it is writen to disk at the given output path.

        :param training_files_path: A string representing the path to the input
        files directory.  
        :param n_iter: An integer representing a number of iterations.  
        :param model_path: A string representing the path to the model to train.  
        :param entities: A list of string representing the entities to consider
        during training.  
        :param best_model_path: A string representing the path to write the best
        trained model.  
        :param max_losses: A float representing the maximum NER losses value
        to consider before start writing best models output.  
        """
        begin_time = datetime.datetime.now()
        onlyfiles = [
            f
            for f in listdir(training_files_path)
            if isfile(join(training_files_path, f))
        ]
        # self.add_new_entity_to_model(entities, model_path)
        best_losses = max_losses
        for file_name in onlyfiles:
            logger.info("Started processing file at \"{}\"...".format(file_name))
            best_losses = self.train_model(
                training_files_path + file_name,
                n_iter,
                model_path,
                entities,
                best_model_path,
                best_losses,
            )
            logger.info("Maximum considerable losses is \"{}\".".format(best_losses))
        diff = datetime.datetime.now() - begin_time
        logger.info("Lasted {} to process {} documents.".format(diff, len(onlyfiles)))

if __name__ == "__main__":
    fire.Fire(SpacyUtils)
