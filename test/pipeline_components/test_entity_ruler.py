import unittest
from spacy.tokens import Span
from test.support.env_case import ModelSetup
import random
import string


def random_string_generator(str_size, allowed_chars):
    return "".join(random.choice(allowed_chars) for x in range(str_size))


class EntityRulerTest(unittest.TestCase):
    def setUp(self):
        pipeline = ["entity_ruler", "entity_matcher", "entity_custom"]
        self.nlp = ModelSetup(pipeline)

    def test_an_entity_ruler_does_find_NOMBRE_ARCHIVO(self):
        chars = string.ascii_letters
        test_sentence = "En el marco de la causa necesitamos verificar el archivo {} y su veracidad"
        extensions = [
            "jpg",
            "png",
            "gif",
            "bmp",
            "tiff",
            "svg",
            "doc",
            "docx",
            "odt",
            "txt",
            "pdf",
            "mp3",
            "avi",
            "mp4",
            "mkv",
            "mpg",
            "mpeg",
            "mov",
            "asf",
            "webm",
            "3gp",
            "3g2",
            "m4v",
        ]
        for ext in extensions:
            filename = random_string_generator(10, chars) + "." + ext
            target_span = (filename, 10, 11)
            doc = self.nlp(test_sentence.format(filename))

            for d in doc.ents:
                a_like_archivo_span = Span(doc, target_span[1], target_span[2], "NOMBRE_ARCHIVO")
                self.assertEqual(a_like_archivo_span.text, target_span[0])
                self.assertIn(a_like_archivo_span, doc.ents)


if __name__ == "__main__":
    unittest.main()
