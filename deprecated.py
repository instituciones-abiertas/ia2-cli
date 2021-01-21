"""
  All functions in here are not being used. Mind that references to other functions / files may not be right.
"""

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
  onlyfiles = [f for f in listdir(training_files_path) if isfile(join(training_files_path, f))]
  random.shuffle(onlyfiles)
  self.add_new_entity_to_model(entities, model_path)
  best_losses = max_losses
  processed_docs = 0
  for file_name in onlyfiles:
    logger.info(f"Started processing file at \"{file_name}\"...")
    best_losses = self.train_model(
      training_files_path + file_name,
      n_iter,
      model_path,
      entities,
      best_model_path,
      best_losses,
    )
    processed_docs = processed_docs + 1
    print(f"Processed {processed_docs} out of {len(onlyfiles)} documents.")
    logger.info(f"Maximum considerable losses is \"{best_losses}\".")
    logger.info((f"Processed {processed_docs} out of {len(onlyfiles)} documents."))
  
  diff = datetime.datetime.now() - begin_time
  logger.info(f"Lasted {diff} to process {len(onlyfiles)} documents.")


def convert_dataturks_to_training_cli(self, input_files_path: str, entities: list, output_file_path: str):
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
    input_files = [f for f in listdir(input_files_path) if isfile(join(input_files_path, f))]

    for input_file in input_files:
        logger.info(f'Extracting raw data and occurrences from file: "{input_file}"...')
        extracted_data = convert_dataturks_to_spacy(f"{input_files_path}/{input_file}", entities)
        TRAIN_DATA = TRAIN_DATA + extracted_data
        logger.info(f'Finished extracting data from file "{input_file}".')

    diff = datetime.datetime.now() - begin_time
    logger.info(f"Lasted {diff} to extract dataturks data from {len(input_files)} documents.")
    logger.info(
        f"Converting {len(TRAIN_DATA)} Documents with Occurences extracted from {len(input_files)} files into Spacy supported format..."
    )

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
                    "matches_text": text[start_idx:end_idx],
                }
                logger.critical(
                    f'Conflicted entity: could not save an entity because it does not match an entity in the given document. Output: "{conflicted_entity}".'
                )
            else:
                new_ents.append(span)

        doc.ents = new_ents
        docs.append(doc)

    diff = datetime.datetime.now() - begin_time
    logger.info(f"Finished Converting {len(TRAIN_DATA)} Spacy Documents into trainable data. Lasted: {diff}")

    try:
        logger.info(f'💾 Writing final output at "{output_file_path}"...')
        srsly.write_json(output_file_path, [docs_to_json(docs)])
        logger.info("💾 Done.")
    except Exception:
        logging.exception(f'An error occured writing the output file at "{output_file_path}".')


def convert_dataturks_to_spacy(self, input_file_path: str, output_file_path: str, entities: list):
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
    logger.info(f'Starts converting data from "{input_file_path}"...')
    training_data = []
    with open(output_file_path, "a+"):
        training_data.append(convert_dataturks_to_spacy(input_file_path, entities))
    with open(output_file_path, "wb") as output:
        pickle.dump(training_data, output, pickle.HIGHEST_PROTOCOL)
    logger.info(f'Succesfully converted data at "{output_file_path}".')

def count_examples(self, files_path: str, entities: list):
  """
  Given the path to a dataturks .json format input file directory and a
  list of entities, prints the total number of examples by label.

  :param files_path: Directory pointing to dataturks .json files to
  be converted.
  :param entities: A list of entities, separated by comma, to be
  considered in the final output.
  """
  entities = entities.split(",")
  input_files_dir_path = files_path
  onlyfiles = [f for f in listdir(input_files_dir_path) if isfile(join(input_files_dir_path, f))]
  all_entities = {}
  for entity in entities:
      entity_length = 0
      for file_ in onlyfiles:
          validation_data = convert_dataturks_to_spacy(input_files_dir_path + file_, [entity])
          for _, annotations in validation_data:
              occurences = annotations.get("entities")
              entity_length = entity_length + len(occurences)
      all_entities[entity] = entity_length
  logger.info(f"Total entities output: {all_entities}")

def show_text(self, files_path: str, entity: str, context_words=0):
  """
  Given the path to a dataturks .json format input file directory and an
  entity name, prints the annotation text from label.

  :param files_path: Directory pointing to dataturks .json files
  :param entity: entity label name.
  """
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
                          else:
                              output = text
                          texts.append(output)
  for text in texts:
      print(text)