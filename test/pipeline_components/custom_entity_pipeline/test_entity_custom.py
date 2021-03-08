import spacy
import unittest

from pipeline_components.entity_custom import EntityCustom, period_rules
from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
    first_left_nbors,
    second_left_nbors,
    first_right_nbors,
)
from pipeline_components.entity_ruler import ruler_patterns
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from test.support.env_case import get_model_path


class EntityCustomTest(unittest.TestCase):
    def setUp(self):
        # Loads a Spacy model
        self.nlp = spacy.load(get_model_path())
        # Adds pipelines
        ruler = EntityRuler(self.nlp, overwrite_ents=True)
        ruler.add_patterns(ruler_patterns)
        self.nlp.add_pipe(ruler)
        entity_matcher = EntityMatcher(self.nlp, matcher_patterns)
        self.nlp.add_pipe(entity_matcher)
        entity_custom = EntityCustom(self.nlp)
        self.nlp.add_pipe(entity_custom)

    def test_a_custom_entity_pipeline_detects_periods(self):
        base_test_senteces = [
            (
                "seis",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de seis {}, si se tratare de un instrumento privado",
            ),
            (
                "6",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de 6 {}, si se tratare de un instrumento privado",
            ),
            (
                "67985",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de 67985 {}, si se tratare de un instrumento privado",
            ),
            (
                "veintitrés",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de veintitrés {}, si se tratare de un instrumento privado",
            ),
        ]
        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            for nbor_word in period_rules:
                test_sentence = base_test_sentece_text.format(nbor_word)
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "seis {nbor}"
                a_like_num_span = Span(doc, target_span_start, target_span_end + 1, "PERIODO")
                expected_period = f"{target_span_text} {nbor_word}"
                self.assertEqual(a_like_num_span.text, expected_period)
                # Asserts a PERIODO like span exists in the document entities
                self.assertIn(a_like_num_span, doc.ents)


if __name__ == "__main__":
    unittest.main()
