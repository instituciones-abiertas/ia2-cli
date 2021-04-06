import unittest

from pipeline_components.entity_matcher import (
    page_first_left_nbors,
    page_second_left_nbors,
    measure_unit_first_right_nbors,
)
from spacy.tokens import Span
from test.support.env_case import ModelSetup


class EntityMatcherTest(unittest.TestCase):
    def setUp(self):
        self.validation_spans = [
            (5, 11, "tres mil ochocientos noventa y nueve"),
            (19, 22, "mil trecientas doce"),
            (34, 35, "134"),
            (42, 47, "cien mil setenta y cinco"),
            (54, 55, "siete"),
            (
                70,
                84,
                "cuatrocientos noventa y tres millones quinientos cuarenta y tres mil seiscientos sesenta y seis",
            ),
            (97, 106, "novecientos noventa y nueve mil cuatrocientos sesenta y seis"),
        ]
        self.test_sentence = "inciso trece, apartado de tres mil ochocientos noventa y nueve. En cuanto a las cauciones, en mil trecientas doce oportunidades solicitan se le imponga a ECorp, en virtud de sus 134 condiciones personales, una caución real de cien mil setenta y cinco pesos ($ 100.020.-) para asegurar siete veces su comparecencia al proceso. De la misma manera se exige respaldo por los cuatrocientos noventa y tres millones quinientos cuarenta y tres mil seiscientos sesenta y seis inmuebles adquiridos en la última década, período en el cual se registraron novecientos noventa y nueve mil cuatrocientos sesenta y seis denuncias relacionadas."
        pipeline = ["entity_ruler", "entity_matcher"]
        self.nlp = ModelSetup(pipeline)

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
        for nbor_word in page_first_left_nbors:
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
        for nbor_word in page_second_left_nbors:
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
        for nbor_word in measure_unit_first_right_nbors:
            test_sentence = base_test_sentece.format(nbor_word)
            doc = self.nlp(test_sentence)
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text "210"
            a_like_num_span = Span(doc, target_span[1], target_span[2], "NUM")
            self.assertEqual(a_like_num_span.text, target_span[0])
            # Asserts there is no num span in this doc entities
            self.assertNotIn(a_like_num_span, doc.ents)

    def test_an_entity_matcher_detects_numbers(self):
        doc = self.nlp(self.test_sentence)
        for span_start_i, span_end_i, span_text in self.validation_spans:
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text 'span_text'
            expected_span = Span(doc, span_start_i, span_end_i, label="NUM")
            self.assertEqual(expected_span.text, span_text)
            # Asserts there is no num span in this doc entities
            self.assertIn(expected_span, doc.ents)
            # Asserts no other num ents have been found
            num_ents = list(filter(lambda e: e.label_ == "NUM", doc.ents))
            self.assertEqual(len(num_ents), len(self.validation_spans))


if __name__ == "__main__":
    unittest.main()
