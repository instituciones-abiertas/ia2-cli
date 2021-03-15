from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.lang.es.lex_attrs import _num_words
from spacy.util import filter_spans


def filter_longer_spans(spans, *, seen_tokens=set(), preserve_spans=[]):
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.

    spans (iterable): The spans to filter.
    RETURNS (list): The filtered spans.
    """

    def get_sort_key(span):
        return (span.end - span.start, -span.start)

    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = preserve_spans
    _seen_tokens = seen_tokens
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in _seen_tokens and span.end - 1 not in _seen_tokens:
            result.append(span)
        _seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


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
    """

    name = "generic_matcher"

    def __init__(self, nlp, matcher_patterns=[]):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        # Adds patterns to the Matcher pipeline
        for entity_label, pattern in matcher_patterns:
            self.matcher.add(entity_label, [pattern], on_match=None)

    def __call__(self, doc):
        matches = self.matcher(doc)
        matched_spans = [Span(doc, start, end, self.nlp.vocab.strings[match_id]) for match_id, start, end in matches]
        # Creates a set of seen tokens so that the filter_longer_spans function
        # prioritizes those spans we are sending.
        seen_tokens = set()
        merged_matched_spans = filter_spans(matched_spans)
        for span in merged_matched_spans:
            seen_tokens.update(range(span.start, span.end))
        doc_ents = merged_matched_spans + list(doc.ents)
        # Merges adjacent entities and removes overlapped entities
        doc.ents = filter_longer_spans(doc_ents, seen_tokens=seen_tokens, preserve_spans=merged_matched_spans)
        return doc
