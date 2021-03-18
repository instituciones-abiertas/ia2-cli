from pipeline_components.entity_custom import (
    EntityCustom,
    law_left_nbors,
    period_rules,
    license_plate_left_nbor,
    address_first_left_nbors,
    address_second_left_nbors,    
)
from pipeline_components.entity_matcher import (
    EntityMatcher,
    matcher_patterns,
    first_left_nbors,
    second_left_nbors,
    first_right_nbors,
)
from pipeline_components.entity_ruler import ruler_patterns
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from test.support.env_case import ModelSetup

import itertools
import spacy
import unittest


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
                # Asserts a LEY span exists in the document entities
                self.assertIn(expected_span, doc.ents)


    def test_a_custom_entity_pipeline_detects_license_plates_entities(self):
        base_test_senteces = [
            (
                "AAA 410",
                23,
                25,
                "Tal situación tuvo lugar, en circunstancias en que ambos se encontraban en el interior del vehículo marca Renault Trafic, {license_plate_nbor} colocado {license_plate}."
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
                test_sentence = base_test_sentece_text.format(license_plate_nbor=nbor_word, license_plate=target_span_text)
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
                "En lo demás, resolví estar a las medidas cautelares impuestas en sede Civil en los términos de la Ley 26485 en protección de la víctima ({article_marked_as_license_plate} CPP)."
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
                "El titular es Mariano Casas con domicilio registrado en la {address}, localidad de Villa Elisa, Provincia de Buenos Aires."
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
                "La acusada fue ubicada en {address} con dos armas cortantes y un teléfono celular robado"
            ),
            (
                "Peribubuy 123",
                5,
                7,
                "La acusada fue ubicada en {address} con dos armas cortantes y un teléfono celular robado"
            ),            
            (
                "Parcela 4",
                15,
                17, #FIXME no incluye depto "Parcela 4 Dpto. E",
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


if __name__ == "__main__":
    unittest.main()
