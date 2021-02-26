import spacy
import unittest

from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
    first_left_nbors, second_left_nbors, first_right_nbors)
from pipeline_components.entity_ruler import ruler_patterns
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from test.support.env_case import get_model_path

class EntityMatcherTest(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load(get_model_path())
        ruler = EntityRuler(self.nlp, overwrite_ents=True)
        ruler.add_patterns(ruler_patterns)
        self.nlp.add_pipe(ruler)
        entity_matcher = EntityMatcher(self.nlp, matcher_patterns)
        self.nlp.add_pipe(entity_matcher)

    def test_an_entity_matcher_does_not_clean_num_entities_when_no_nbors_match(self):
        target_span = ("veinte", 6, 7)
        test_sentece = "En el marco del aniversario número {} del club queremos destacar los trabajos y los procesos de construcción que se fueron haciendo."
        doc = self.nlp(test_sentece.format(target_span[0]))
        # Checks that the text is tokenized the way we expect, so that we
        # can correctly pick up a span with text "veinte" in positions 6 and 7
        a_like_num_span = Span(doc, target_span[1], target_span[2], "NUM")
        self.assertEqual(a_like_num_span.text, target_span[0])
        # Asserts there is a num span in this doc entities
        self.assertIn(a_like_num_span, doc.ents)

    def test_an_entity_matcher_cleans_out_first_left_nbor_num_entities(self):
        target_span = ("210", 12, 13)
        base_test_sentece = "Tipo de audiencia: Audiencia de admisibilidad de la prueba ({} 210 Código Procesal Penal de la CABA, en adelante CPPCABA)."
        for nbor_word in first_left_nbors:
            test_sentence = base_test_sentece.format(nbor_word)
            doc = self.nlp(test_sentence)
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text "210"
            a_like_num_span = Span(doc, target_span[1], target_span[2], "NUM")
            self.assertEqual(a_like_num_span.text, target_span[0])
            # Asserts there is no num span in this doc entities
            self.assertNotIn(a_like_num_span, doc.ents)

    def test_an_entity_matcher_cleans_out_second_left_nbor_num_entities(self):
        target_span = ("210", 13, 14)
        base_test_sentece = "Tipo de audiencia: Audiencia de admisibilidad de la prueba ({}. 210 Código Procesal Penal de la CABA, en adelante CPPCABA)."
        for nbor_word in second_left_nbors:
            test_sentence = base_test_sentece.format(nbor_word)
            doc = self.nlp(test_sentence)
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text "210"
            a_like_num_span = Span(doc, target_span[1], target_span[2], "NUM")
            self.assertEqual(a_like_num_span.text, target_span[0])
            # Asserts there is no num span in this doc entities
            self.assertNotIn(a_like_num_span, doc.ents)

    def test_an_entity_matcher_cleans_out_first_right_nbor_num_entities(self):
        target_span = ("210", 12, 13)
        base_test_sentece = "Evidencia en Audiencia de admisibilidad para la prueba (Se encuentra a 210 {}, establecido en el Código Procesal Penal de la CABA, en adelante CPPCABA)."
        for nbor_word in first_right_nbors:
            test_sentence = base_test_sentece.format(nbor_word)
            doc = self.nlp(test_sentence)
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text "210"
            a_like_num_span = Span(doc, target_span[1], target_span[2], "NUM")
            self.assertEqual(a_like_num_span.text, target_span[0])
            # Asserts there is no num span in this doc entities
            self.assertNotIn(a_like_num_span, doc.ents)

if __name__ == "__main__":
    unittest.main()
