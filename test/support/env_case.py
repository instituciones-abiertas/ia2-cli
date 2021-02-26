import os
import spacy
from spacy.pipeline import EntityRuler
from pipeline_components.entity_ruler import ruler_patterns
from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
)


def get_model_path():
    """
    Simple retrieval function.
    Returns MODEL_FILE or raises OSError.
    """
    model_path = os.getenv("TEST_MODEL_FILE", default=None)
    if model_path is None:
        raise OSError("TEST_MODEL_FILE environment is not set.")
    return model_path


def setup_model():
    """
    Retrieve model with current pipeline
    Returns NLP
    """
    nlp = spacy.load(get_model_path())
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(ruler_patterns)
    nlp.add_pipe(ruler)
    entity_matcher = EntityMatcher(nlp, matcher_patterns)
    nlp.add_pipe(entity_matcher)
    return nlp
