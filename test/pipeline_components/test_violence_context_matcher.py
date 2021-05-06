from pipeline_components.entity_custom import (
    EntityCustom,
)
from pipeline_components.entity_matcher import (
    ArticlesMatcher,
    EntityMatcher,
    ViolenceContextMatcher,
    gender_violence_types,
    matcher_patterns,
    violence_nbors,
    violence_types,
    fetch_cb_by_tag,
)
from pipeline_components.entity_ruler import fetch_ruler_patterns_by_tag
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from test.support.data_case import generate_fake_sentences
from test.support.env_case import ModelSetup

import itertools
import spacy
import unittest


class ViolenceContextMatcherTest(unittest.TestCase):
    def setUp(self):
        # Loads a Spacy model
        nlp = ModelSetup()
        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns(fetch_ruler_patterns_by_tag("todas"))
        nlp.add_pipe(ruler)

        entity_matcher = EntityMatcher(
            nlp,
            matcher_patterns,
            after_callbacks=[cb(nlp) for cb in fetch_cb_by_tag("todas")],
        )
        nlp.add_pipe(entity_matcher)
        entity_custom = EntityCustom(nlp)
        nlp.add_pipe(entity_custom)
        self.nlp = nlp

    def test_a_custom_entity_pipeline_detects_violence_contexts(self):
        entity_left_nbors = violence_nbors + list(map(lambda nbor: f"{nbor}:", violence_nbors))
        # We take away a few elements of the remaining lists to avoid running very long tests
        _two_words_violence = list(
            map(lambda words: f"{words[0]} y {words[1]}", itertools.combinations(violence_types, 2))
        )
        two_words_violence = list(itertools.islice(_two_words_violence, int(len(_two_words_violence) / 3)))
        three_words_violence = list(
            itertools.islice(
                map(lambda words: f"{words[0]}, {words[1]} y {words[2]}", itertools.combinations(violence_types, 3)),
                len(two_words_violence),
            )
        )
        # A list of tuple where the first component is an article value and the
        # second the value length in spacy terms.
        entity_values = list(
            map(lambda v: (v, len(self.nlp(v))), violence_types + two_words_violence + three_words_violence)
        )
        base_test_sentences = [
            (
                26,
                "Lo mismo Gloria Estefan, que es amiga de la denunciante, conoce toda la situación entre las partes, incluso se refirió a la {entity_left_nbor} {entity_value} sufrida por Falcón Ramón, y dijo que incluso le vio moretones",
            ),
            (
                23,
                "Esto da cuenta de la situación de vulnerabilidad de la víctima, por lo cual debe enmarcarse dentro de un contexto de {entity_left_nbor} {entity_value} y de protección integral de las mujeres.",
            ),
            (
                48,
                "No sólo del hecho puntual relativo al despojo de la damnificada de la posesión que ejercía sobre el inmueble en cuestión, sino también del contexto de violencia en el que estaba inmersa, que conforme los testimonios detallados, se dan al menos dos tipos de {entity_left_nbor} {entity_value}. Por su parte, el Defensor intentó rebatir la hipótesis acusatoria.",
            ),
        ]
        test_sentences = generate_fake_sentences(entity_values, base_test_sentences)

        for (entity_value, (target_span_start, target_span_end, base_test_sentece_text)) in test_sentences:
            for left_nbor_word in entity_left_nbors:
                left_nbor_word_length = len(self.nlp(left_nbor_word))
                test_sentence = base_test_sentece_text.format(
                    entity_left_nbor=left_nbor_word, entity_value=entity_value
                )
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "{left_nbor_word} {nbor}"
                expected_span = Span(
                    doc,
                    target_span_start - 1,
                    target_span_end + (left_nbor_word_length - 1),
                    label="CONTEXTO_VIOLENCIA",
                )
                self.assertEqual(expected_span.text, f"{left_nbor_word} {entity_value}")
                # Asserts a CONTEXTO_DE_VIOLENCIA span exists in the document entities
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_gender_violence_contexts(self):
        entity_left_nbors = violence_nbors + list(map(lambda nbor: f"{nbor}:", violence_nbors))
        # A list of tuple where the first component is an article value and the
        # second the value length in spacy terms.
        parsed_gender_violence_types = list(map(lambda word: f"de {word}", gender_violence_types))
        entity_values = list(map(lambda v: (v, len(self.nlp(v))), parsed_gender_violence_types))
        base_test_sentences = [
            (
                26,
                "Lo mismo Gloria Estefan, que es amiga de la denunciante, conoce toda la situación entre las partes, incluso se refirió a la {entity_left_nbor} {entity_value} sufrida por Falcón Ramón, y dijo que incluso le vio moretones",
            ),
            (
                11,
                "Intervienen con asesoramiento y contención a personas de víctima de {entity_left_nbor} {entity_value}. Intervino porque le ingresan las actuaciones por la denuncia.",
            ),
            (
                23,
                "Esto da cuenta de la situación de vulnerabilidad de la víctima, por lo cual debe enmarcarse dentro de un contexto de {entity_left_nbor} {entity_value} y de protección integral de las mujeres.",
            ),
            (
                48,
                "No sólo del hecho puntual relativo al despojo de la damnificada de la posesión que ejercía sobre el inmueble en cuestión, sino también del contexto de violencia en el que estaba inmersa, que conforme los testimonios detallados, se dan al menos dos tipos de {entity_left_nbor} {entity_value}. Por su parte, el Defensor intentó rebatir la hipótesis acusatoria.",
            ),
        ]
        test_sentences = generate_fake_sentences(entity_values, base_test_sentences)

        for (entity_value, (target_span_start, target_span_end, base_test_sentece_text)) in test_sentences:
            for left_nbor_word in entity_left_nbors:
                left_nbor_word_length = len(self.nlp(left_nbor_word))
                test_sentence = base_test_sentece_text.format(
                    entity_left_nbor=left_nbor_word, entity_value=entity_value
                )
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "{left_nbor_word} {nbor}"
                expected_span = Span(
                    doc,
                    target_span_start - 1,
                    target_span_end + (left_nbor_word_length - 1),
                    label="CONTEXTO_VIOLENCIA_DE_GÉNERO",
                )
                self.assertEqual(expected_span.text, f"{left_nbor_word} {entity_value}")
                # Asserts a CONTEXTO_VIOLENCIA_DE_GÉNERO span exists in the document entities
                self.assertIn(expected_span, doc.ents)


if __name__ == "__main__":
    unittest.main()
