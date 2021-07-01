from pipeline_components.entity_custom import (
    EntityCustom,
    law_left_nbors,
    period_rules,
)
from pipeline_components.entity_matcher import (
    ArticlesMatcher,
    EntityMatcher,
    matcher_patterns,
    article_left_nbors,
)
from pipeline_components.entity_ruler import fetch_ruler_patterns_by_tag
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from test.support.data_case import generate_fake_sentences
from test.support.env_case import ModelSetup

import spacy
import unittest


class ArticleMatcherTest(unittest.TestCase):
    def setUp(self):
        # Loads a Spacy model
        nlp = ModelSetup()
        ruler = EntityRuler(nlp, overwrite_ents=True)
        ruler.add_patterns(fetch_ruler_patterns_by_tag("todas"))
        nlp.add_pipe(ruler)
        articles_matcher = ArticlesMatcher(nlp)
        entity_matcher = EntityMatcher(nlp, matcher_patterns, after_callbacks=[articles_matcher])
        nlp.add_pipe(entity_matcher)
        entity_custom = EntityCustom(nlp)
        nlp.add_pipe(entity_custom)
        self.nlp = nlp

    def test_an_article_matcher_detects_articles(self):
        article_values = [
            # A tuple where the first component is an article value and the second the value length in spacy terms.
            ("20", 1),
            ("20 y 21", 3),
            ("20, 21, 22, 23", 7),
            ("20, 21, 22, 23 y 24", 9),
            ("13000, 13001, 13002, 13003, 13004, 13005, 13006, 13007 y 13420", 17),
        ]
        base_test_senteces = [
            (
                # The index where the article value starts at
                5,
                "De aplicación supletoria conforme {article_nbor} {article_value}), en virtud de su leyes a número 134 condiciones personales.",
            ),
            (
                8,
                "Ha sufrido una modificación relativamente reciente mediante {article_nbor} {article_value} en función de la cual no cualquier persona puede ser autora de esta contravención.",
            ),
            (
                6,
                "Schumacher, Michael s/{article_nbor} {article_value} violar clausura impuesta por autoridad judicial o administrativa.",
            ),
            (
                20,
                "NO HACER LUGAR a la excepción por manifiesto defecto en la pretensión por atipicidad interpuesta por la defensa ({article_nbor} {article_value} inciso c) y 197 CPPCABA.",
            ),
        ]
        test_sentences = generate_fake_sentences(article_values, base_test_senteces)

        for (article_value, (target_span_start, target_span_end, base_test_sentece_text)) in test_sentences:
            for left_nbor_word in article_left_nbors:
                test_sentence = base_test_sentece_text.format(article_nbor=left_nbor_word, article_value=article_value)
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "{left_nbor_word} {nbor}"
                expected_span = Span(doc, target_span_start - 1, target_span_end, label="ARTÍCULO")
                self.assertEqual(expected_span.text, f"{left_nbor_word} {article_value}")
                # Asserts an ARTÍCULO span exists in the document entities
                self.assertIn(expected_span, doc.ents)


if __name__ == "__main__":
    unittest.main()
