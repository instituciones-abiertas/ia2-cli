import os
import spacy
from spacy.pipeline import EntityRuler
from pipeline_components.entity_ruler import fetch_ruler_patterns_by_tag
from pipeline_components.entity_custom import EntityCustom
from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
)


class ModelSetup:
    def __new__(self, pipe_names=[]):
        self.valid_pipe_names = ["entity_ruler", "entity_matcher", "entity_custom"]
        self.remove = self.valid_pipe_names
        self.pipe_names = pipe_names
        self.model_path = self.get_model_path()
        self.ruler_patterns = fetch_ruler_patterns_by_tag("todas")
        return self.setup_model(self)

    def get_model_path():
        """
        Simple retrieval function.
        Returns MODEL_FILE or raises OSError.
        """
        model_path = os.getenv("TEST_MODEL_FILE", default=None)
        if model_path is None:
            raise OSError("TEST_MODEL_FILE environment is not set.")
        return model_path

    def remove_pipelines(self):
        for pipe in self.remove:
            if pipe in self.nlp.pipe_names:
                self.nlp.remove_pipe(pipe)

    def check_pipeline_names(self):
        for pipe in self.pipe_names:
            if pipe not in self.valid_pipe_names:
                raise ValueError(f"{pipe} is not on valid test pipelines")

    def setup_model(self):
        """
        Retrieve model with custom pipeline
        Returns NLP
        """
        self.check_pipeline_names(self)
        self.nlp = spacy.load(self.get_model_path())
        self.remove_pipelines(self)

        if "entity_ruler" in self.pipe_names:
            ruler = EntityRuler(self.nlp, overwrite_ents=True)
            ruler.add_patterns(self.ruler_patterns)
            self.nlp.add_pipe(ruler)

        if "entity_matcher" in self.pipe_names:
            entity_matcher = EntityMatcher(self.nlp, matcher_patterns)
            self.nlp.add_pipe(entity_matcher)

        if "entity_custom" in self.pipe_names:
            entity_custom = EntityCustom(self.nlp)
            self.nlp.add_pipe(entity_custom)

        return self.nlp
