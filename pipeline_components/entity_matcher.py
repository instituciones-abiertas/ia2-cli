from spacy.matcher import Matcher
from spacy.tokens import Span

matcher_patterns = {
    "NUM": [{"LIKE_NUM": True}],
}


def exist_n_token(token_id, n, maximo):
    indice = token_id + n
    return indice > 0 and indice <= maximo


def not_in_nbor(maximo, ent, ent_name, word_list, nborPos):
    # el vecino no tienen una palabra que pertenezca a la lista dada
    return (
        ent.label_ == ent_name
        and exist_n_token(ent[0].i, nborPos, maximo)
        and ent[0].nbor(nborPos).text not in word_list
    )


def overlap(span, span_list):
    for s in span_list:
        if (span.start >= s.start and span.start < s.end) or (s.start >= span.start and s.end <= span.end):
            return True
    return False


class EntityMatcher(object):
    name = "entity_matcher"

    def __init__(self, nlp, matcher_patterns=matcher_patterns):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        # El mactch_id debe ser igual a el NOMBRE de la entidad
        for match_id, pattern in matcher_patterns.items():
            self.matcher.add(match_id, None, pattern)

    def __call__(self, doc):
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            maximo = len(doc)
            label = self.nlp.vocab.strings[match_id]
            span = Span(doc, start, end, label)

            first_left_nbor = not_in_nbor(
                maximo,
                span,
                "NUM",
                [
                    "página",
                    "pag",
                    "p",
                    "p.",
                    "pág",
                    "pág.",
                    "fs",
                    "art",
                    "art.",
                    "arts",
                    "arts.",
                    "inciso",
                    "artículo",
                    "artículos",
                    "inc",
                ],
                -1,
            )
            second_left_nbor = not_in_nbor(
                maximo,
                span,
                "NUM",
                ["página", "pag", "p", "pág", "fs", "art", "arts", "inciso", "artículo", "artículos", "inc"],
                -2,
            )
            first_right_nbor = not_in_nbor(
                maximo, span, "NUM", ["inc", "hs", "horas", "metros", "m", "gr", "grs", "gramos", "km", "kg", "cm"], 1
            )
            if (first_left_nbor or second_left_nbor or first_right_nbor) and not overlap(span, doc.ents):
                doc.ents = list(doc.ents) + [span]
        return doc
