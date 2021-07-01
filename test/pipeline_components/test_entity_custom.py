from pipeline_components.entity_custom import (
    EntityCustom,
    law_left_nbors,
    period_rules,
    license_plate_left_nbor,
    address_first_left_nbors,
    address_second_left_nbors,
    age_right_token,
    age_text_in_token,
    actuacion_number_indicator,
    actuacion_nbor_token,
    number_abreviated_indicator,
    expediente_indicator,
)
from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
    page_first_left_nbors,
    page_second_left_nbors,
    measure_unit_first_right_nbors,
)
from spacy.tokens import Span
from test.support.env_case import ModelSetup

import itertools
import spacy
import unittest


"""
Consider that Spacy parses 4 words on the sides of each token to understand its context,
so when creating tests we should use texts as close as possible to how they appear in the court judgments.
"""


class EntityCustomTest(unittest.TestCase):
    def setUp(self):
        # Loads a Spacy model
        pipeline = ["entity_ruler", "entity_matcher", "entity_custom"]
        self.nlp = ModelSetup(pipeline)

    def test_a_custom_entity_pipeline_detects_periods(self):
        base_test_senteces = [
            (
                "seis",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de seis {}, si se tratare de un instrumento privado",
            ),
            (
                "6",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de 6 {}, si se tratare de un instrumento privado",
            ),
            (
                "67985",
                11,
                12,
                "Si se tratare de un instrumento público y con prisión de 67985 {}, si se tratare de un instrumento privado",
            ),
            (
                "veintitrés mil quinientos",
                11,
                14,
                "Si se tratare de un instrumento público y con prisión de veintitrés mil quinientos {}, si se tratare de un instrumento privado",
            ),
            (
                "tres mil ochocientos noventa y nueve",
                11,
                17,
                "Si se tratare de un instrumento público y con prisión de tres mil ochocientos noventa y nueve {}, si se tratare de un instrumento privado",
            ),
        ]
        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            for nbor_word in period_rules:
                test_sentence = base_test_sentece_text.format(nbor_word)
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "seis {nbor}"
                a_like_num_span = Span(doc, target_span_start, target_span_end + 1, label="PERIODO")
                expected_period = f"{target_span_text} {nbor_word}"
                self.assertEqual(a_like_num_span.text, expected_period)
                # Asserts a PERIODO span exists in the document entities
                self.assertIn(a_like_num_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_law_entities(self):
        nums = ["5845", "5666", "6", "12"]
        base_test_senteces = [
            (
                10,
                11,
                "Ha sufrido una modificación relativamente reciente mediante la {law_nbor} número {law_num}, en función de la cual no cualquier persona puede ser autora de esta contravención",
            ),
            (
                24,
                25,
                "Schumacher, Michael s/art. 73 Violar clausura impuesta por autoridad judicial o administrativa (Art. 74 según TC {law_nbor} número {law_num} y modif.)",
            ),
            (
                41,
                42,
                "NO HACER LUGAR a la excepción por manifiesto defecto en la pretensión por atipicidad interpuesta por la defensa (arts. 195, inciso c) y 197 CPPCABA -de aplicación supletoria conforme el art. 6 de la {law_nbor} número {law_num}).",
            ),
        ]

        test_sentences = list(itertools.product(nums, base_test_senteces))

        for (law_num, (target_span_start, target_span_end, base_test_sentece_text)) in test_sentences:
            for left_nbor_word in law_left_nbors:
                test_sentence = base_test_sentece_text.format(law_nbor=left_nbor_word, law_num=law_num)
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "seis {nbor}"
                expected_span = Span(doc, target_span_start, target_span_end, label="LEY")
                self.assertEqual(expected_span.text, law_num)
                # Asserts a PERIODO span exists in the document entities
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detect_loc_entities(self):
        locs = ["YPF", "504"]
        base_test_senteces = [
            (
                4,
                6,
                "Existio un allanamiento en paraje {loc_nbor} donde se encontraron estupefacientes ",
            ),
            (
                12,
                14,
                "Existe un procedimiento a llevarse a cabo dentro del radio de el asentamiento {loc_nbor} de la periferia de la ciudad",
            ),
            (
                12,
                14,
                "Existe un procedimiento a llevarse a cabo dentro del radio de la localidad {loc_nbor} de la periferia de la ciudad",
            ),
            (
                24,
                26,
                "En la mañana Fierro, Martin s/art. 23 Inculpar a vecino por desacato judicial o administrativa en la actualidad ubicada en country {loc_nbor}",
            ),
        ]

        test_sentences = list(itertools.product(locs, base_test_senteces))

        for (left_nbor_word, (target_span_start, target_span_end, base_test_sentece_text)) in test_sentences:

            test_sentence = base_test_sentece_text.format(loc_nbor=left_nbor_word)

            doc = self.nlp(test_sentence)

            expected_span = Span(doc, target_span_start, target_span_end, label="LOC")
            # Check if span detected in doc.ents
            self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detect_false_positive_loc_entities(self):
        # Primer test para chequear casos de falsos positivios en PER no esten incluidos en las ocurrencias detectadas
        locs = ["Moreno", "Fierro"]
        base_test_sentences = [
            (
                6,
                8,
                10,
                12,
                "En las personas de  Juan Antonio Barrio {loc_name} y mariela barrio {loc_name} se encontraron estupefacientes ",
            ),
            (6, 8, 10, 12, "Acompañada de otras personas como Roxana villa {loc_name}, Martín Villa {loc_name}  "),
        ]

        test_sentences = list(itertools.product(locs, base_test_sentences))
        for (
            right_loc_word,
            (
                target_span_start,
                target_span_end,
                another_target_span_start,
                another_target_span_end,
                base_test_sentece_text,
            ),
        ) in test_sentences:

            test_sentence = base_test_sentece_text.format(loc_name=right_loc_word)

            doc = self.nlp(test_sentence)

            expected_span = Span(doc, target_span_start, target_span_end, label="LOC")
            another_expected_span = Span(doc, another_target_span_start, another_target_span_end, label="LOC")
            # Filtrado solo entidades del tipo LOC
            # onlyLOCents = list(filter(lambda ent: ent.label_ == "LOC", doc.ents))
            self.assertNotIn(expected_span, doc.ents)
            self.assertNotIn(another_expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_license_plates_entities(self):
        base_test_senteces = [
            (
                "AAA 410",
                23,
                25,
                "Tal situación tuvo lugar, en circunstancias en que ambos se encontraban en el interior del vehículo marca Renault Trafic, {license_plate_nbor} colocado {license_plate}.",
            ),
            (
                "AC 154 BC",
                4,
                7,
                "El vehículo {license_plate_nbor}: {license_plate} fue encontrado prendido fuego en la intersección de las Avenidas 1 y 2.",
            ),
            (
                "AC154BC",
                9,
                10,
                "Schumacher, Michael circulaba en el vehículo con {license_plate_nbor} {license_plate} a altas velocidades mientras tiraba objetos desde su vehículo.",
            ),
            (
                "110 ABC",
                3,
                5,
                "{license_plate_nbor} del vehículo {license_plate}, fue visto por última vez el 24 de mayo de 1996.",
            ),
        ]

        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            for nbor_word in license_plate_left_nbor:
                test_sentence = base_test_sentece_text.format(
                    license_plate_nbor=nbor_word, license_plate=target_span_text
                )
                doc = self.nlp(test_sentence)
                # Checks that the text is tokenized the way we expect, so that we
                # can correctly pick up a span with text "{nbor} AAA 410"
                expected_span = Span(doc, target_span_start, target_span_end, label="PATENTE_DOMINIO")
                self.assertEqual(expected_span.text, target_span_text)
                # Asserts a PATENTE_DOMINIO span exists in the document entities
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_removes_articles_marked_as_license_plates_entities(self):
        base_test_senteces = [
            (
                "174bis",
                27,
                28,
                "En lo demás, resolví estar a las medidas cautelares impuestas en sede Civil en los términos de la Ley 26485 en protección de la víctima ({article_marked_as_license_plate} CPP).",
            ),
        ]

        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            test_sentence = base_test_sentece_text.format(article_marked_as_license_plate=target_span_text)
            doc = self.nlp(test_sentence)
            # Checks that the text is tokenized the way we expect, so that we
            # can correctly pick up a span with text "174bis"
            expected_span = Span(doc, target_span_start, target_span_end, label="ART")
            self.assertEqual(expected_span.text, target_span_text)
            # Asserts a ART span exists in the document entities
            self.assertNotIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_address_entities(self):
        base_test_sentences = [
            (
                "calle 44, nro 62",
                10,
                15,
                "El titular es Mariano Casas con domicilio registrado en la {address}, localidad de Villa Elisa, Provincia de Buenos Aires.",
            ),
            (
                "Vieytes 1690",
                9,
                11,
                "dispuesta en sede administrativa respecto del inmueble sito en {address} y Av. Osvaldo Cruz 2010,",
            ),
            (
                "calle José León Suárez 4686",
                3,
                8,
                "Fijar domicilio en {address} de esta Ciudad, y comunicar a la Fiscalía el cambio de éste.",
            ),
            (
                "pasaje Pernambuco 2244",
                24,
                27,
                "Realizar diez horas de tareas comunitarias en la Asociación Civil Centro Comunitario y Social Comedor y Merendero “Justos Somos Más” ubicada en {address}, de esta ciudad",
            ),
            (
                "Cristomo Alvarez 461",
                11,
                14,
                "El Sr. Cristian Gomez, con último domicilio en la calle {address} de esta Ciudad; para que una vez que sea encontrado, sea detenido",
            ),
            (
                "Av. Chaco 655",
                10,
                13,
                "acercándose al domicilio de su ex pareja, con dirección {address}, de esta Ciudad",
            ),
            (
                "Angel Peribubuy 123",
                5,
                8,
                "La acusada fue ubicada en {address} con dos armas cortantes y un teléfono celular robado",
            ),
            (
                "Peribubuy 123",
                5,
                7,
                "La acusada fue ubicada en {address} con dos armas cortantes y un teléfono celular robado",
            ),
            (
                "Parcela 4",
                15,
                17,  # FIXME no incluye depto "Parcela 4 Dpto. E",
                "Fijar residencia en el domicilio de su madre ubicado en el Barrio La Boca 12 {address} de esta ciudad ",
            ),
        ]

        for address, target_span_start, target_span_end, base_test_sentece_text in base_test_sentences:
            for first_left_nbor_word in address_first_left_nbors:
                test_sentence = base_test_sentece_text.format(address=address)
                doc = self.nlp(test_sentence)
                expected_span = Span(doc, target_span_start, target_span_end, label="DIRECCIÓN")
                self.assertEqual(expected_span.text, address)
                # Asserts a DIRECCIÓN span exists in the document entities
                # print(f"doc.ents {doc.ents}")
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detext_fecha_resolucion(self):
        # 1) La primera fecha que encuentra si esta dentro de los primeros 100 tokens entonces lo transforma a FECHA_RESOLUCION
        # 2) Si hay más de una fecha, la segunda fecha no tiene que ser FECHA_RESOLUCION, sino que es FECHA
        # 3) Igual a 1 pero no al inicio
        # 4) Si se detecta luego de los 100 tokens no es una FECHA_RESOLUCION, sino que es una FECHA
        fecha = "9 de julio de 2020"
        base_test_senteces = [
            (3, 8, "Buenos Aires, {fecha}", "FECHA_RESOLUCION"),
            (9, 14, "Buenos Aires, {fecha} y {fecha}", "FECHA"),
            (
                63,
                68,
                "RESOLUCION INTERLOCUTORIA \n No hace lugar a pedido de allanamiento y requisa personal por carácter prematuro de la medida (art. 108 y 112 CPPCABA; arts. 18 y 75 inc. 22 CN; art. 9 DADDH; art. 12 DUDH; art. 112 CADH; 138 y 123 CCABA).,\n Buenos Aires, {fecha}",
                "FECHA_RESOLUCION",
            ),
            (
                123,
                128,
                "RESOLUCION INTERLOCUTORIA \n No hace lugar a pedido de allanamiento y requisa personal por carácter prematuro de la medida (art. 108 y 112 CPPCABA; arts. 18 y 75 inc. 22 CN; art. 9 DADDH; art. 12 DUDH; art. 112 CADH; 138 y 123 CCABA).,\n RESOLUCION INTERLOCUTORIA \n No hace lugar a pedido de allanamiento y requisa personal por carácter prematuro de la medida (art. 108 y 112 CPPCABA; arts. 18 y 75 inc. 22 CN; art. 9 DADDH; art. 12 DUDH; art. 112 CADH; 138 y 123 CCABA).,\n Buenos Aires, {fecha}",
                "FECHA",
            ),
        ]

        for start, end, sentence, label in base_test_senteces:
            test_sentence = sentence.format(fecha=fecha)
            doc = self.nlp(test_sentence)
            expected_span = Span(doc, start, end, label=label)
            self.assertEqual(expected_span.text, fecha)
            self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_age(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_case_number(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_cuij_number(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_actuacion_number(self):
        base_test_senteces = [
            (
                "15612345/2020",
                7,
                8,
                "CUIJ: IAP J-11-10022223-1/2020-0 {actuacion_nbor_token} {actuacion_number_indicator}: {nro_actuacion}",
            ),
            (
                "23456101/2019",
                11,
                12,
                "Número: COU 7111/2020-0 CUIJ: CAU J-11-11111114-5/2020-0 {actuacion_nbor_token} {actuacion_number_indicator}: {nro_actuacion}",
            ),
        ]

        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            for nbor_word in actuacion_nbor_token:
                test_sentence = base_test_sentece_text.format(
                    actuacion_nbor_token=actuacion_nbor_token,
                    actuacion_number_indicator=actuacion_number_indicator,
                    nro_actuacion=target_span_text,
                )
                doc = self.nlp(test_sentence)
                expected_span = Span(doc, target_span_start, target_span_end, label="NUM_ACTUACIÓN")
                self.assertEqual(expected_span.text, target_span_text)
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_expediente_number(self):
        base_test_senteces = [
            (
                "15612/20",
                7,
                8,
                "Para resolver en el presente {expediente_indicator} {number_abreviated_indicator} {nro_expediente}, del registro de la Secretaría General de la Cámara de fuero",
            ),
            (
                "1234/19",
                17,
                18,
                "Refiere que solo tiene dos oposiciones, en cuanto a la incorporación por lectura del {expediente_indicator} civil {nro_expediente} del Juzgado Civil N° 126, en cuanto la Fiscalía requirió la incorporación in totum.",
            ),
            (
                "4321/19",
                18,
                19,
                "Refiere que solo tiene dos oposiciones, en cuanto a la incorporación por lectura del {expediente_indicator} penal {number_abreviated_indicator} {nro_expediente} del Juzgado Civil N° 45, en cuanto la Fiscalía requirió la incorporación in totum.",
            ),
        ]

        for target_span_text, target_span_start, target_span_end, base_test_sentece_text in base_test_senteces:
            for nbor_word in expediente_indicator:
                test_sentence = base_test_sentece_text.format(
                    expediente_indicator=expediente_indicator,
                    number_abreviated_indicator=number_abreviated_indicator,
                    nro_expediente=target_span_text,
                )
                doc = self.nlp(test_sentence)
                expected_span = Span(doc, target_span_start, target_span_end, label="NUM_EXPEDIENTE")
                self.assertEqual(expected_span.text, target_span_text)
                self.assertIn(expected_span, doc.ents)

    def test_a_custom_entity_pipeline_detects_judge(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_secretary(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_prosecutor(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_ombuds_person(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_accused(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_advisor(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_ip_address(self):
        print("TODO: not implemented")
        pass

    def test_a_custom_entity_pipeline_detects_phone(self):
        print("TODO: not implemented")
        pass


if __name__ == "__main__":
    unittest.main()
