from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.lang.es.lex_attrs import _num_words
from spacy.util import filter_spans


def repeat_patterns(patterns, times):
    """
    Utility function that receives a pattern to return a list that contains
    that pattern multiplied by times. The final length of the list is equal to
    len(patterns) * times.
    """
    generated_patterns = []
    for i in range(0, times):
        [generated_patterns.append(pattern) for pattern in patterns]
    return generated_patterns


class GenericMatcher(object):
    """
    GenericMatcher: Given an NLP instance, and list of patterns, generates a
    pipeline that matches tokens against each of those patterns to return an
    updated Doc object.

    A matches_priority option may be given to set how those overlapped spans
    should be treated.
    """

    name = "generic_matcher"

    def __init__(self, nlp, matcher_patterns=[], *, matches_priority="preserve"):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.matches_priority = matches_priority
        # Adds patterns to the Matcher pipeline
        for entity_label, pattern in matcher_patterns:
            self.matcher.add(entity_label, [pattern], on_match=None)

    def __call__(self, doc):
        matches = self.matcher(doc)
        matched_spans = [Span(doc, start, end, self.nlp.vocab.strings[match_id]) for match_id, start, end in matches]
        _doc_ents = []
        if self.matches_priority == "override":
            _doc_ents = matched_spans + list(doc.ents)
        elif self.matches_priority == "preserve":
            _doc_ents = list(doc.ents) + matched_spans
        # Merges adjacent entities and removes overlapped entities
        doc.ents = filter_spans(_doc_ents)
        return doc
