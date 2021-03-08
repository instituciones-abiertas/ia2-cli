from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.lang.es.lex_attrs import _num_words
from spacy.util import filter_spans

num_words = _num_words + [
    "quinientos",
    "quinientas",
    "ochocientos",
    "ochocientas",
]

first_left_nbors = [
    "página",
    "pag",
    "p",
    "p.",
    "pág",
    "pág.",
    "fs",
    "art",
    "arts",
    "inciso",
    "articulo",
    "artículo",
    "artículos",
    "articulos",
    "inc",
]
second_left_nbors = [
    "página",
    "pag",
    "fs",
    "art",
    "arts",
    "inciso",
    "articulo",
    "artículo",
    "artículos",
    "articulos",
    "inc",
]
first_right_nbors = ["inc", "metros", "m", "gr", "grs", "gramos", "km", "kg", "cm"]


def exist_n_token(token_index, nbor_position, document_length):
    """
    Returns true if a token neighbor exists in a document.

    :param token_index: The document index of a token
    :param nbor_position: An integer that represents a number of characters to
    the left (negative values) or to the right (positive values)
    :param document_length: An integer that represents the number of tokens in a
    document
    """
    index = token_index + nbor_position
    return index >= 0 and index < document_length


def not_in_nbor(document_length, span, ent_name, word_list, nbor_position):
    """
    Given a span, returns true whenever then span tokene exists in the document
    and does not match a string in a word list

    :param document_length: The number of tokens in a document
    :param span: A span object to evaluate
    :param ent_name: An entity name
    :param word_list: A list of allowed words
    :param nbor_position: An integer representing a number of characters to the
    left (negative values) or to the right (positive values)
    """
    if exist_n_token(span[0].i, nbor_position, document_length):
        return (
            span.label_ == ent_name
            and span[0].nbor(nbor_position).text not in word_list
        )
    return True


def is_start_of_span_contained(span, target_span):
    return span.start >= target_span.start and span.start < target_span.end


def is_end_of_span_contained(span, target_span):
    return target_span.start >= span.start and target_span.end <= span.end


def overlaps(span, span_list):
    """
    Returns True if the given span, or part of it, is contained in a given span
    list.
    """
    for s in span_list:
        if is_start_of_span_contained(span, s) or is_end_of_span_contained(span, s):
            return True
    return False


matcher_patterns = [
    # Multi-num tokens
    [
        {"LOWER": {"IN": num_words}, "OP": "+"},
        {"ORTH": 'y', "OP": "*"},
        {"LOWER": {"IN": num_words}, "OP": "+"}
    ],
    # Single num tokens
    [{"LOWER": {"IN": num_words}, "OP": "+"}],
    # Digit-like words
    [{"IS_DIGIT": True}],
]


class EntityMatcher(object):
    """
    EntityMatcher: matches contexts around "NUM" entities in a Document and
    cleans them out of "NUM" labels.
    """

    name = "entity_matcher"

    def __init__(self, nlp, matcher_patterns=matcher_patterns):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        # Adds patterns to the Matcher pipeline
        for pattern in matcher_patterns:
            self.matcher.add("NUM", [pattern], on_match=None)

    def __call__(self, doc):
        matches = self.matcher(doc)

        matched_spans = [Span(doc, start, end, self.nlp.vocab.strings[match_id]) for match_id, start, end in matches]
        filtered_spans = filter_spans(matched_spans)

        for span in filtered_spans:
            document_length = len(doc)

            def does_not_have_first_left_nbor(document_length, span):
                return not_in_nbor(
                    document_length,
                    span,
                    "NUM",
                    first_left_nbors,
                    -1,
                )

            def does_not_have_second_left_nbor(document_length, span):
                return not_in_nbor(
                    document_length,
                    span,
                    "NUM",
                    second_left_nbors,
                    -2,
                )

            def does_not_have_first_right_nbor(document_length, span):
                return not_in_nbor(document_length, span, "NUM", first_right_nbors, 1)

            if (
                does_not_have_first_left_nbor(document_length, span)
                and does_not_have_second_left_nbor(document_length, span)
                and does_not_have_first_right_nbor(document_length, span)
            ) and not overlaps(span, doc.ents):
                doc.ents = list(doc.ents) + [span]
            else:
                # FIXME this one ent has been discarded by the nbor word lists.
                # We should consider assigning them to an entity, or filter them
                # somewhere else
                print(f"[FIXME] Should process this span as another ent: `{span}`")

        return doc
