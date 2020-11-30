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
from spacy.gold import GoldParse
from os import listdir
from os.path import isfile, join

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DROPOUT_RATE = 0.2  ## Configuracion del set de dropout del entrenamiento


def removeEntitiesNotInList(spacyfile, entityList):
    # Esta funcion remueve de una training_data preparada para spacy ,todas las entidades que estan en la lista
    file = spacyfile
    newTrainingData = []

    for hit in file:
        entities = []
        data = hit[1]["entities"]
        for ent in data:
            if ent[2] in entityList:
                # print("Se conservo la entidad {}".format(ent[2]))
                entities.append(ent)
            else:
                print("Se descarto la entidad {}".format(ent[2]))
        newTrainingData.append((hit[0], {"entities": entities}))
    return newTrainingData


def convert_dataturks_to_spacy(dataturks_JSON_FilePath, entityList):
    try:
        training_data = []
        lines = []

        with open(dataturks_JSON_FilePath, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                text = data["content"]
                entities = []
                annotations = []
                if data["annotation"]:
                    for annotation in data["annotation"]:
                        point = annotation["points"][0]
                        label = annotation["label"]
                        if label[0] in entityList:
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
                    count_common = 0
                    count_overlaped = 0
                    for annotation in annotations:

                        start = annotation[0]
                        end = annotation[1]
                        labels = annotation[2]
                        if start not in seen_tokens and end - 1 not in seen_tokens:
                            count_common = count_common + 1
                            seen_tokens.update(range(start, end))
                            if isinstance(labels, list):
                                labels = labels[0]
                            entities.append((start, end + 1, labels))
                        else:
                            count_overlaped = count_overlaped + 1
                            print(
                                "{} {} {} esta overlapeada".format(start, end, labels)
                            )
                training_data.append((text, {"entities": entities}))

        print("Entidades normales : {}".format(count_common))
        print("Entidades overlopeadas : {}".format(count_overlaped))
        return training_data
    except Exception as e:
        logging.exception(
            "Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e)
        )
        return None


class SpacyConverterTrainer:
    """
    Convertidor de formato Dataturks a Spacy y eliminador de repetidos .

    Metodos disponibles:
    create_blank_model(path_save_model: str)--> Crea un modelo en blanco
    create_custom_spacy_model(spacy_model: str, path_save_model: str)--> crea un modelo partiendo de un modelo de  spacy
    add_new_entity_to_model(ents: list, model_path: str)--> Agrega nuevas entidades para entrenar al modelo
    train_model(
        path_data_training: str, n_iter: int, model_path: str, ents: list
    )--> Entrena un modelo n_iter veces con la data en formato dataturks pasada.
    """

    def create_blank_model(self, path_save_model: str):
        """
        Crea un modelo en blanco .
        :param path_save_model: path donde se guardar el modelo

        """
        nlp = spacy.blank("es")
        nlp.to_disk(path_save_model)
        logger.info("Modelo creado exitosamente {path_save_model}...")

    def create_custom_spacy_model(self, spacy_model: str, path_save_model: str):
        """
        Crea un modelo en base al modelo pasado por parametro .
        :param spacy_model: modelo de spacy a partir de base
        :param path_save_model: path donde se guardar el modelo

        """
        nlp = spacy.load(spacy_model)
        nlp.to_disk(path_save_model)
        logger.info("Modelo creado exitosamente {path_save_model}...")

    def add_new_entity_to_model(self, ents: list, model_path: str):
        """
        Agrega nuevas entidades al modelo para entrenar
        :param ents: lista de entidades a entrenar
        :param model_path: path del modelo a  utilizar

        """

        nlp = spacy.load(model_path)
        ner = nlp.pipe_names
        if not ner:
            component = nlp.create_pipe("ner")
            nlp.add_pipe(component)
        ner = nlp.get_pipe("ner")
        for ent in ents:
            ner.add_label(ent)
        print(ner.move_names)
        nlp.to_disk(model_path)

        logger.info("Agregados exitosamente las entidades al {model_path}...")

    def convert_dataturks_to_spacy(self, convert_data_path: str, output_path: str):
        """
        Dado una data en formato dataturks, la transforma para formato spacy.
        :param convert_data_path: path del output de dataturks
        :param output_path: path del archivo resultante

        """
        logger.info(f"Loading convert data from {convert_data_path} ...")
        training_data = []
        log = convert_dataturks_to_spacy(convert_data_path)
        print(log)
        with open(output_path, "a+") as f:
            training_data.append(convert_dataturks_to_spacy(convert_data_path))
        with open(output_path, "wb") as output:
            pickle.dump(training_data, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Informacion convertida exitosamente")

    def train_model(
        self,
        path_data_training: str,
        n_iter: int,
        model_path: str,
        ents: list,
        path_best_model: str,
        best_losses: float,
    ):
        """
        Dado una data en formato dataturks, la transforma para formato spacy.
        :param path_data_training: path de la info a entrenar
        :param n_iter: Numero de cantidad de veces que se va correr el scripts
        :param model_path: path del modelo a  utilizar
        :ents: lista de entidades a entrenar
        """
        best = best_losses

        training_data = convert_dataturks_to_spacy(path_data_training, ents)

        nlp = spacy.load(model_path)

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        ner = nlp.get_pipe("ner")

        for _, annotations in training_data:

            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # only train NER
        with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module="spacy")
            nlp.begin_training()
            # print(training_data)

            for itn in range(n_iter):
                random.shuffle(training_data)  # Se randomiza
                losses = {}
                # Crea mini paquetes
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                #

                for batch in batches:
                    texts, annotations = zip(*batch)
                    # Codigo para probar bilout_tags
                    # doc = nlp(texts[0])
                    # entities = annotations[0]
                    # tags = biluo_tags_from_offsets(doc, entities)
                    # print(tags)
                    print(
                        "Se estan cargando las siguientes entidades para entrenar: {}".format(
                            annotations
                        )
                    )

                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=DROPOUT_RATE,
                        losses=losses,
                    )
                print(losses)
                try:
                    numero_losses = losses.get("ner")
                    print(type(numero_losses))
                    print(type(best))
                    if numero_losses < best and numero_losses > 0:
                        best = numero_losses
                        nlp.to_disk(path_best_model)
                        print("Guarde el modelo con este losses {}".format(best))

                except Exception:
                    print("Batch sin entidades entrenadas")

            # save model to output directory
            nlp.to_disk(model_path)
            print(
                "El mejor losses fue {} y esta guardado el modelo en {}".format(
                    best, path_best_model
                )
            )
            return best

    def get_entities(self, model_path: str, text: str):
        """
        Dado una data en formato dataturks, la transforma para formato spacy.
        :param model_path:ruta del modelo
        :param text: texto a traducir.
        """
        nlp = spacy.load(model_path, disable=["tagger", "parser"])
        doc = nlp(text)
        spacy.displacy.serve(doc, style="ent", page=True, port=5030)

    def scorer_model(self, model_path: str, text: str, annotations: list):
        scorer = Scorer()
        print(scorer.scores)
        nlp = spacy.load(model_path)
        try:
            doc_gold_text = nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annotations)
            pred_value = nlp(text)
            scorer.score(pred_value, gold)
            print(scorer.scores)
        except Exception as e:
            print(e)

    def all_files_in_folder(
        self,
        path_folder: str,
        n_iter: int,
        model_path: str,
        ents: list,
        path_best_model: str,
    ):
        """
        Entrenar un modelo con todos los archivos en la carpeta actual.
        :n_iter:numero de iteraciones
        :param model_path:ruta del modelo
        :param ents: Lista de las entidades a anonimizar.
        """
        begin_time = datetime.datetime.now()
        onlyfiles = [f for f in listdir(path_folder) if isfile(join(path_folder, f))]
        # self.add_new_entity_to_model(ents, model_path)
        best_losses = 100.0
        for file in onlyfiles:
            print("Se esta procesando {}".format(file))
            best_losses = self.train_model(
                path_folder + file,
                n_iter,
                model_path,
                ents,
                path_best_model,
                best_losses,
            )
            print(best_losses)
        diff = datetime.datetime.now() - begin_time
        print("Tardo {} en procesar la info".format(diff))


if __name__ == "__main__":
    fire.Fire(SpacyConverterTrainer)