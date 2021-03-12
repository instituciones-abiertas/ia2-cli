from spacy.tokens import Span
from spacy.util import filter_spans
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

law_left_nbors = [
    "ley",
    "leyes",
]

address_first_left_nbors = ["calle", "Calle", "dirección", "Dirección",
                    "avenida", "av.", "Avenida", "Av.", 
                    "pasaje", "Pasaje", "Parcela", "parcela"]
address_second_left_nbors = [
    "instalación", "contramano", "sita", "sitas", "sito", "sitos",
    "real", "domiciliado", "domiciliada", "constituido",
    "constituida", "contramano", "intersección", "domicilio",
    "ubicado", "registrado", "ubicada", "real"
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


def is_law(ent):
    first_token = ent[0]
    return ent.label_ == "NUM" and (
        first_token.nbor(-1).lower_ in law_left_nbors
        or first_token.nbor(-2).lower_ in law_left_nbors
        or first_token.nbor(-3).lower_ in law_left_nbors
    )


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
    last_token = ent[len(ent) - 1]
    return ent.label_ in ["NUM"] and last_token.nbor(1).text in period_rules


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
    first_token = ent[0]
    last_token = ent[-1]
    is_address_1_tokens_to_left = first_token.nbor(-1).lower_ in address_first_left_nbors
    is_address_2_tokens_to_left = first_token.nbor(-2).lower_ in address_second_left_nbors
    try:
        is_address_3_tokens_to_left = first_token.nbor(-3).lower_ in address_first_left_nbors
    except:
        is_address_3_tokens_to_left = False
    try:
        is_address_4_tokens_to_left = first_token.nbor(-4).lower_ in address_first_left_nbors
    except:
        is_address_4_tokens_to_left = False

    is_address = ent.label_ in ["NUM"] and (
        is_address_1_tokens_to_left
        or is_address_2_tokens_to_left
        or first_token.nbor(-2).lower_ in address_first_left_nbors
        or is_address_3_tokens_to_left
        or is_address_4_tokens_to_left
    )
    if is_address:
        print(f"ent: {ent}   ent.label_: {ent.label_}")
        print(f"first_token.nbor(-1) {first_token.nbor(-1).lower_}")
        print(f"first_token.nbor(-2) {first_token.nbor(-2).lower_}")
        print(f"first_token.nbor(-3) {first_token.nbor(-3).lower_}")
        print(f"first_token.nbor(-4) {first_token.nbor(-4).lower_}")
    return (ent.label_ in ["PER"] and (
        is_address_1_tokens_to_left
        or is_address_2_tokens_to_left
        or last_token.like_num
        or last_token.nbor().like_num
    ) or is_address)


def get_aditional_left_tokens_for_address(ent):
    if ent.label_ in ["NUM"]:
        token = ent[0]
        if token.nbor(-1).lower_ in address_first_left_nbors:
            return 1
        if token.nbor(-2).lower_ in address_second_left_nbors or token.nbor(-2).lower_ in address_first_left_nbors:
            return 2
        if token.nbor(-3).lower_ in address_first_left_nbors:
            return 3
        if token.nbor(-4).lower_ in address_first_left_nbors:
            return 4
    return 0


#def is_numeric_address(ent):
#    left_nbors = ["calle", "Calle", "dirección", "Dirección", "hasta", "domicilio", "Domicilio"]
#    nro_nbors = ["nro", "numero", "número"]
#    # TODO considerar agregar "nro", "número" como left_nbors y armar una sola etiqueta DIRECCION
#
#    token = ent[0]
##    if ent.label_ == "NUM":
##      print(f"ent {ent}")
##      print(f"ent.label_ {ent.label_}")
##      print(f"ent.text {ent.text}")
##      print(f"token.nbor(-1) {token.nbor(-1).lower_}")
##      print(f"token.nbor(-2) {token.nbor(-2).lower_}")
#    is_number_of_address = False
#    try:
#        #      print(f"token.nbor(-3) {token.nbor(-3).lower_}")
#        #      print(f"token.nbor(-4) {token.nbor(-4).lower_}")
#        is_number_of_address = (token.nbor(-1).lower_ in nro_nbors or token.nbor(-2).lower_ in nro_nbors) and (
#            token.nbor(-3).lower_ in left_nbors or token.nbor(-4).lower_ in left_nbors)
##      print(f"is_number_of_address: {is_number_of_address}")
#    except:
#        print("\n")
#
#    return ent.label_ in ["NUM"] and ((
#        token.nbor(-1).lower_ in left_nbors
#    ) or is_number_of_address)


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
            if not is_from_first_tokens(ent.start) and is_law(ent):
                new_ents.append(Span(doc, ent.start, ent.end, "LEY"))
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
                #FIXME se pueden unificar los token adicionales?
                token_adicional = 1 if ent[-1].nbor().like_num else 0
                address_token = get_aditional_left_tokens_for_address(ent)
                #print(f"ent: {ent} - address_token {address_token}")
                if address_token > 1:
                  new_ent_start = ent.start - address_token #+ 1

                  #TODO se podría usar remove_wrong_labeled_entity_span() que está en el branch de mejora_patente
                  span_to_remove_index = None
                  for i, new_ent in enumerate(new_ents):
                    if new_ent_start >= new_ent.start and new_ent_start <= new_ent.end or ent.end >= new_ent.start and ent.end <= new_ent.end:
                      span_to_remove_index = i
                      break

                  if span_to_remove_index:
                    #print(f"ANTES new_ents {new_ents}")
                    filtered = new_ents.pop(span_to_remove_index)
                    #print(f"DESP POP new_ents {new_ents}")
                    #print(f"ent.end - new_ent_start = {ent.end - new_ent_start}")
                    #print(f"filtered.end - filtered.start = {filtered.end - filtered.start}")
                    if (ent.end - new_ent_start) > (filtered.end - filtered.start):
                      new_ents.append(Span(doc, new_ent_start, ent.end, label="DIRECCIÓN"))
                    else:
                      new_ents.append(filtered)
                    #print(f"DESPUES new_ents {new_ents}")
                  else:
                    new_ents.append(Span(doc, new_ent_start, ent.end, label="DIRECCIÓN"))
                    #print(f"DESPUES new_ents {new_ents}")
                else:
                  new_ents.append(Span(doc, ent.start, ent.end + token_adicional, label="DIRECCIÓN"))
                
            #if not is_from_first_tokens(ent.start) and is_numeric_address(ent):
            #    new_ents.append(Span(doc, ent.start - 1, ent.end, label="DIRECCIÓN"))                
            if not is_from_first_tokens(ent.start) and is_ip_address(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_IP"))
            if not is_from_first_tokens(ent.start) and is_phone(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_TELÉFONO"))

        if new_ents:
            # We'd always want the new entities to be appended first because
            # filter_spans prioritizes the first occurrences on overlapping
            doc.ents = filter_spans(new_ents + list(doc.ents))

        return doc
