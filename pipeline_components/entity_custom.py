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

license_plate_left_nbor = [
    "patente",
    "dominio",
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

def get_token_in_x_left_pos(token, pos, nbors):
    try:
        return token.nbor(-pos).lower_ in nbors
    except:
        return False

def is_address(ent):
    first_token = ent[0]
    last_token = ent[-1]
    is_address_1_tokens_to_left = get_token_in_x_left_pos(first_token, 1, address_first_left_nbors)
    is_address_2_tokens_to_left_first_nbors = get_token_in_x_left_pos(first_token, 2, address_first_left_nbors)
    is_address_2_tokens_to_left_second_nbors = get_token_in_x_left_pos(first_token, 2, address_second_left_nbors)
    is_address_3_tokens_to_left = get_token_in_x_left_pos(first_token, 3, address_first_left_nbors)
    is_address_4_tokens_to_left = get_token_in_x_left_pos(first_token, 4, address_first_left_nbors)

    is_address_from_PER = ent.label_ in ["PER"] and (
        is_address_1_tokens_to_left
        or is_address_2_tokens_to_left_second_nbors
        or last_token.like_num
        or last_token.nbor().like_num
    )

    is_address_from_NUM = ent.label_ in ["NUM"] and (
        is_address_1_tokens_to_left
        or is_address_2_tokens_to_left_second_nbors
        or is_address_2_tokens_to_left_first_nbors
        or is_address_3_tokens_to_left
        or is_address_4_tokens_to_left
    )
    if is_address_from_NUM:
        print(f"ent: {ent}   ent.label_: {ent.label_}")
        print(f"first_token.nbor(-1) {first_token.nbor(-1).lower_}")
        print(f"first_token.nbor(-2) {first_token.nbor(-2).lower_}")
        print(f"first_token.nbor(-3) {first_token.nbor(-3).lower_}")
        print(f"first_token.nbor(-4) {first_token.nbor(-4).lower_}")

    return is_address_from_PER or is_address_from_NUM


def get_aditional_left_tokens_for_address(ent):
    if ent.label_ in ["PER"] and ent[-1].nbor().like_num:
        return 1
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


def could_be_an_article(ent):
    # TODO deberíamos centralizar esta extracción de tokens según posición
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    second_left_token = token.nbor(-2).lower_
    third_left_token = token.nbor(-3).lower_
    dont_consider = "bis"

    return (
        ent.label_ == "PATENTE_DOMINIO"
        and token.lower_.find(dont_consider) != -1
        and first_left_token not in license_plate_left_nbor
        and second_left_token not in license_plate_left_nbor
        and third_left_token not in license_plate_left_nbor
    )


def is_license_plate(ent):
    # TODO deberíamos centralizar esta extracción de tokens según posición
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    second_left_token = token.nbor(-2).lower_
    third_left_token = token.nbor(-3).lower_

    return token.like_num and (
        first_left_token in license_plate_left_nbor
        or second_left_token in license_plate_left_nbor
        or third_left_token in license_plate_left_nbor
    )


def get_start_end_license_plate(ent):
    # TODO deberíamos centralizar esta extracción de tokens según posición
    token = ent[0]
    first_left_token = token.nbor(-1).lower_
    first_right_token = token.nbor(1).lower_
    if len(ent.text) != 3:  # this means it is not an "incomplete" license plate
        return ent.start, ent.end
    if len(first_left_token) == 3 and isinstance(first_left_token, str):
        # 3 letras - 3 núm
        return ent.start - 1, ent.end
    if (
        len(first_left_token) == 2
        and len(first_right_token) == 2
        and isinstance(first_left_token, str)
        and isinstance(first_right_token, str)
    ):
        # 2 letras - 3 núm - 2 letras
        return ent.start - 1, ent.end + 1
    if len(first_right_token) == 3 and isinstance(first_right_token, str):
        # 3 núm - 3 letras
        return ent.start, ent.end + 1


def remove_wrong_labeled_entity_span(ent_list, ent_to_remove):
    return [ent for ent in ent_list if not (ent_to_remove.start == ent.start and ent_to_remove.end == ent.end)]


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
            if not is_last(ent.start, doc) and is_period(ent):
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
                address_token = get_aditional_left_tokens_for_address(ent)
                new_ent_start = ent.start - address_token

                ent_to_remove = None
                for i, new_ent in enumerate(new_ents):
                    if new_ent_start >= new_ent.start and new_ent_start <= new_ent.end or ent.end >= new_ent.start and ent.end <= new_ent.end:
                        ent_to_remove = new_ent
                        break

                if ent_to_remove:
                    new_ents = remove_wrong_labeled_entity_span(new_ents, ent_to_remove)
                    #print(f"ent.end - new_ent_start = {ent.end - new_ent_start}")
                    #print(f"filtered.end - filtered.start = {filtered.end - filtered.start}")
                    if (ent.end - new_ent_start) > (ent_to_remove.end - ent_to_remove.start):
                        print(f"agregoooo una nueva")
                        new_ents.append(Span(doc, new_ent_start, ent.end, label="DIRECCIÓN"))
                    else:
                        print(f"agregoooo la ent_to_remove")
                        #TODO debería encontrar cuando pasa por acá, cuando detecta algo que borró pero tiene que volver a agregar?
                        new_ents.append(ent_to_remove)
                else:
                    new_ents.append(Span(doc, new_ent_start, ent.end, label="DIRECCIÓN"))
                
            #if not is_from_first_tokens(ent.start) and is_numeric_address(ent):
            #    new_ents.append(Span(doc, ent.start - 1, ent.end, label="DIRECCIÓN"))                
            if not is_from_first_tokens(ent.start) and is_ip_address(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_IP"))
            if not is_from_first_tokens(ent.start) and is_phone(ent):
                new_ents.append(Span(doc, ent.start, ent.end, label="NUM_TELÉFONO"))
            if not is_from_first_tokens(ent.start) and could_be_an_article(ent) and ent.label_ == "PATENTE_DOMINIO":
                doc.ents = remove_wrong_labeled_entity_span(doc.ents, ent)
            if not is_from_first_tokens(ent.start) and is_license_plate(ent):
                start, end = get_start_end_license_plate(ent)
                new_ents.append(Span(doc, start, end, label="PATENTE_DOMINIO"))

        if new_ents:
            # We'd always want the new entities to be appended first because
            # filter_spans prioritizes the first occurrences on overlapping
            doc.ents = filter_spans(new_ents + list(doc.ents))

        return doc
