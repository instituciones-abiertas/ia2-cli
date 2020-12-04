import unittest
from os import listdir
from os.path import isfile, join
from train import SpacyConverterTrainer, convert_dataturks_to_spacy

def report_scores(scores):
    """
    prints precision recall and f_measure
    :param scores:
    :return:
    """

    precision = '%.2f' % scores['ents_p']
    recall = '%.2f' % scores['ents_r']
    f_measure = '%.2f' % scores['ents_f']
    print('%-10s %-10s %-10s' % (precision, recall, f_measure))

class TestTraining(unittest.TestCase):

  def test_score(self):
    trainer = SpacyConverterTrainer()
    
    input_files_dir_path = "./data/validation/"
    entities = ["PER", "LOC", "DIRECCIÓN", "OCUPACIÓN/PROFESIÓN", "PATENTE_DOMINIO"]

    model_path = "models/blank/2020-12-03/"

    onlyfiles = [f for f in listdir(input_files_dir_path) if isfile(join(input_files_dir_path, f))]
    for file_ in onlyfiles:
      validation_data = convert_dataturks_to_spacy(input_files_dir_path + file_, entities)
      for input_, annotations in validation_data:
        
        results = trainer.evaluate(model_path, input_, annotations.get("entities"))

        print('%-10s %-10s %-10s' % ('Precision', 'Recall', 'F Measure'))
        report_scores(results)
        print("\n")

    # self.assertEqual('foo'.upper(), 'FOO')

  def test_batch_labels(self):
    input_files_dir_path = "./data/training/"
    entities = ["PER", "LOC", "DIRECCIÓN", "OCUPACIÓN/PROFESIÓN", "PATENTE_DOMINIO", "ARTÍCULO"]
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
    print(all_entities)
    

if __name__ == '__main__':
  unittest.main()