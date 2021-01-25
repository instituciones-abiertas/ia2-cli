from spacy.tokens import Span


def is_age(token, right_token, token_sent):
    return token.like_num and right_token.text == "años" and "edad" in token_sent.text


def is_caseNumber(token, first_left_token, second_left_token, token_sent):
    return token.like_num and (
        (first_left_token.lower_ == "nº" and second_left_token.lower_ == "causa") or first_left_token.lower_ == "caso"
    )


def is_cuijNumber(token):
    return (token.is_ascii and token.nbor(-3).lower_ == "cuij") or (token.like_num and token.nbor(-3).lower_ == "cuij")


def is_actuacionNumber(token):
    return token.nbor(-1).lower_ == ":" and token.nbor(-2).lower_ == "nro" and token.nbor(-3).lower_ == "actuación"


def is_expedienteNumber(token):
    return (
        token.nbor(-1).lower_ == "nº"
        and (token.nbor(-3).lower_ == "expediente" or token.nbor(-2).lower_ == "expediente")
    ) or (token.like_num and token.nbor(-2).lower_ == "expediente")


def is_last(token_id, doc):
    return token_id == len(doc) - 1


def is_from_first_tokens(token_id):
    return token_id <= 2


def is_judge(ent):
    first_token = ent[0]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in ["juez", "Juez"]
        or first_token.nbor(-2).lemma_ in ["juez", "Juez"]
        or first_token.nbor(-2).lemma_ in ["juez", "Juez"]
    )


def is_secretary(ent):
    first_token = ent[0]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in ["secretario", "Secretario"]
        or first_token.nbor(-2).lemma_ in ["secretario", "Secretario"]
        or first_token.nbor(-2).lemma_ in ["secretario", "Secretario"]
    )


def is_prosecutor(ent):
    first_token = ent[0]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in ["fiscal", "Fiscal, Fiscalía, fiscalía"]
        or first_token.nbor(-2).lemma_ in ["fiscal", "Fiscal, Fiscalía, fiscalía"]
        or first_token.nbor(-2).lemma_ in ["fiscal", "Fiscal, Fiscalía, fiscalía"]
    )


def is_address(ent):
    first_left_nbors = ["calle", "Calle", "dirección", "Dirección", "hasta"]
    second_left_nbors = [
        "instalación",
        "contramano",
        "sita",
        "sitas",
        "sito",
        "sitos",
        "real",
        "domiciliado",
        "domiciliada",
        "constituido",
        "constituida",
        "contramano",
        "intersección",
        "domicilio",
        "ubicado",
        "ubicada",
        "real",
    ]
    first_token = ent[0]
    last_token = ent[-1]

    return ent.label_ in ["PER"] and (
        first_token.nbor(-1).lower_ in first_left_nbors
        or first_token.nbor(-2).lower_ in second_left_nbors
        or last_token.like_num
        or last_token.nbor().like_num
    )


def filter_spans(a_list, b_list):
    # filtra spans de a_list que se overlapeen con algun span de b_list
    def overlap(span, span_list):
        for s in span_list:
            if (span.start >= s.start and span.start < s.end) or (s.start >= span.start and s.end <= span.end):
                return True
        return False

    return [span for span in a_list if not overlap(span, b_list)]


class EntityCustom(object):
    name = "entity_custom"

    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, doc):
        new_ents = []
        for token in doc:
            if not is_last(token.i, doc) and is_age(token, token.nbor(1), token.sent):
                new_ents.append(Span(doc, token.i, token.i + 1, label="EDAD"))
            if not is_from_first_tokens(token.i) and is_caseNumber(token, token.nbor(-1), token.nbor(-2), token.sent):
                new_ents.append(Span(doc, token.i, token.i + 1, label="NUM_CAUSA"))
            if not is_from_first_tokens(token.i) and is_cuijNumber(token):
                new_ents.append(Span(doc, token.i, token.i + 1, label="NUM_CUIJ"))
            if not is_from_first_tokens(token.i) and is_actuacionNumber(token):
                new_ents.append(Span(doc, token.i, token.i + 1, label="NUM_ACTUACIÓN"))
            if not is_from_first_tokens(token.i) and is_expedienteNumber(token):
                new_ents.append(Span(doc, token.i, token.i + 1, label="NUM_EXPEDIENTE"))
        for ent in doc.ents:
            if not is_from_first_tokens(ent.start) and is_judge(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="JUEZ"))
            if not is_from_first_tokens(ent.start) and is_secretary(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="SECRETARIX"))
            if not is_from_first_tokens(ent.start) and is_prosecutor(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="FISCAL"))
            if not is_from_first_tokens(ent.start) and is_address(ent):
                token_adicional = 1 if ent[-1].nbor().like_num else 0
                new_ents.append(Span(doc, ent.start, ent.end + token_adicional, label="DIRECCIÓN"))

        if new_ents:
            filtered_ents = filter_spans(doc.ents, new_ents)
            doc.ents = list(filtered_ents) + new_ents

        return doc
