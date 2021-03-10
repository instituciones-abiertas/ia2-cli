from spacy.tokens import Span
import re


period_rules = [
    "segundo",
    "segundos",
    "minuto",
    "minutos",
    "hr",
    "hs",
    "hora",
    "horas",
    "año",
    "años",
    "dia",
    "día",
    "dias",
    "días",
    "mes",
    "meses",
]


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


def is_law(token, first_left_token, second_left_token, third_left_token):
  law_texts = ["ley", "leyes"]
  return token.like_num and (first_left_token.lower_ in law_texts or second_left_token.lower_ in law_texts or third_left_token.lower_ in law_texts)


def is_last(token_id, doc):
    return token_id == len(doc) - 1


def is_from_first_tokens(token_id):
    return token_id <= 2


def is_judge(ent):
    first_token = ent[0]
    judge_lemma = ["juez", "jueza", "Juez", "Jueza"]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in judge_lemma
        or first_token.nbor(-2).lemma_ in judge_lemma
        or first_token.nbor(-3).lemma_ in judge_lemma
    )


def is_period(ent):
    first_token = ent[0]
    return ent.label_ in ["NUM"] and first_token.nbor(1).text in period_rules


def is_secretary(ent):
    first_token = ent[0]
    secretarix_lemma = [
        "secretario",
        "secretaria",
        "prosecretario",
        "prosecretaria",
        "Prosecretario",
        "Prosecretaria",
        "Secretario",
        "Secretaria",
    ]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in secretarix_lemma
        or first_token.nbor(-2).lemma_ in secretarix_lemma
        or first_token.nbor(-3).lemma_ in secretarix_lemma
    )


def is_prosecutor(ent):
    first_token = ent[0]
    prosecutor_lemma = ["fiscal", "fiscalía", "Fiscal", "Fiscalía"]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in prosecutor_lemma
        or first_token.nbor(-2).lemma_ in prosecutor_lemma
        or first_token.nbor(-3).lemma_ in prosecutor_lemma
    )


def is_ombuds_person(ent):
    first_token = ent[0]
    ombuds_person_lemma = ["defensor", "defensora", "Defensora", "Defensor"]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in ombuds_person_lemma
        or first_token.nbor(-2).lemma_ in ombuds_person_lemma
        or first_token.nbor(-3).lemma_ in ombuds_person_lemma
    )


def is_accused(ent):
    first_token = ent[0]
    accused_lemma = [
        "acusado",
        "acusada",
        "imputado",
        "imputada",
        "infractor",
        "infractora",
        "Acusado",
        "Acusada",
        "Imputado",
        "Imputada",
        "Infractor",
        "Infractora",
    ]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in accused_lemma
        or first_token.nbor(-2).lemma_ in accused_lemma
        or first_token.nbor(-3).lemma_ in accused_lemma
    )


def is_advisor(ent):
    first_token = ent[0]
    advisor_lemma = ["asesor", "asesora", "Asesor", "Asesora"]
    return ent.label_ in ["PER", "LOC"] and (
        first_token.nbor(-1).lemma_ in advisor_lemma
        or first_token.nbor(-2).lemma_ in advisor_lemma
        or first_token.nbor(-3).lemma_ in advisor_lemma
    )


def is_ip_address(ent):
    octet_rx = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    pattern = re.compile(r"^{0}(?:\.{0}){{3}}$".format(octet_rx))
    is_ip = pattern.match(str(ent))
    return ent.label_ in ["NUM", "NUM_IP"] and is_ip


def is_phone(ent):
    first_token = ent[0]
    phone_lemma = ["teléfono", "tel", "celular", "número", "numerar", "telefónico"]
    phone_text = ["telefono", "tel", "cel"]
    return ent.label_ == "NUM" and (
        first_token.nbor(-1).lemma_ in phone_lemma
        or first_token.nbor(-2).lemma_ in phone_lemma
        or first_token.nbor(-3).lemma_ in phone_lemma
        or first_token.nbor(-1).text in phone_text
        or first_token.nbor(-2).text in phone_text
        or (first_token.nbor(-1).text == "(" and first_token.nbor(1).text == ")")
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


def could_be_an_article(ent):
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    second_left_token = token.nbor(-2).lower_
    third_left_token = token.nbor(-3).lower_
    license_texts = ["patente", "dominio"]
    dont_consider = "bis"
    
    return ent.label_ == "PATENTE_DOMINIO" and token.lower_.find(dont_consider) != -1 and first_left_token not in license_texts and second_left_token not in license_texts and third_left_token not in license_texts


def is_license_plate(ent):
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    second_left_token = token.nbor(-2).lower_
    third_left_token = token.nbor(-3).lower_
    license_texts = ["patente", "dominio"]

    return token.like_num and (first_left_token in license_texts or second_left_token in license_texts or third_left_token in license_texts)


def get_start_end_license_plate(ent):
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    first_right_token = token.nbor(1).lower_
    if len(ent.text) != 3: #this means it is not an "incomplete" license plate
        return ent.start, ent.end
    if len(first_left_token) == 3 and isinstance(first_left_token, str):
        #3 letras - 3 núm
        return ent.start - 1, ent.end
    if len(first_left_token) == 2 and len(first_right_token) == 2 and isinstance(first_left_token, str) and isinstance(first_right_token, str):
        #2 letras - 3 núm - 2 letras
        return ent.start - 1, ent.end + 1
    if len(first_right_token) == 3 and isinstance(first_right_token, str):
        #3 núm - 3 letras
        return ent.start, ent.end + 1


def remove_wrong_labeled_entity_span(doc, ent):
    found_ent = None
    for i, new_ent in enumerate(doc.ents):
        if ent.start == new_ent.start and ent.end == new_ent.end:
            found_ent = new_ent
            break

    if found_ent:
        filtered_ents = filter_spans(doc.ents, [found_ent])
        doc.ents = list(filtered_ents)    


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
            if not is_from_first_tokens(token.i) and is_law(token, token.nbor(-1), token.nbor(-2), token.nbor(-3)):
                new_ents.append(Span(doc, token.i, token.i + 1, label="LEY"))
        for ent in doc.ents:
            if not is_from_first_tokens(ent.start) and is_period(ent):
                new_ents.append(Span(doc, ent.start, ent.end + 1, label="PERIODO"))
            if not is_from_first_tokens(ent.start) and is_judge(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="JUEZ/A"))
            if not is_from_first_tokens(ent.start) and is_secretary(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="SECRETARIX"))
            if not is_from_first_tokens(ent.start) and is_prosecutor(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="FISCAL"))
            if not is_from_first_tokens(ent.start) and is_ombuds_person(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="DEFENSOR/A"))
            if not is_from_first_tokens(ent.start) and (is_accused(ent) or is_advisor(ent)):
                new_ents.append(Span(doc, ent.start, ent.end, label="PER"))
            if not is_from_first_tokens(ent.start) and is_address(ent):
                token_adicional = 1 if ent[-1].nbor().like_num else 0
                new_ents.append(Span(doc, ent.start, ent.end + token_adicional, label="DIRECCIÓN"))
            if not is_from_first_tokens(ent.start) and is_ip_address(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_IP"))
            if not is_from_first_tokens(ent.start) and is_phone(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_TELÉFONO"))
            if not is_from_first_tokens(ent.start) and could_be_an_article(ent) and ent.label_ == "PATENTE_DOMINIO":
                remove_wrong_labeled_entity_span(doc, ent)

            if not is_from_first_tokens(token.i) and is_license_plate(ent):
                start, end = get_start_end_license_plate(ent)
                new_ents.append(Span(doc, start, end, label="PATENTE_DOMINIO"))                
                

        if new_ents:
            filtered_ents = filter_spans(doc.ents, new_ents)
            doc.ents = list(filtered_ents) + new_ents

        return doc
