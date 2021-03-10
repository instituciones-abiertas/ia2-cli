def generate_fake_sentences(entity_values, test_sentences):
    """
    Given a list of entity values and a list of test sentences, returns the
    product of those list to return a mix of cases between sentences and entity
    values.

    Input list structure:

    * A list of tuple where the first element is an entity value and the second
    the length of that value in SpaCy terms. E.g.:

    a NUM entity value
    [("setenta y ocho", 3), ("setenta mil quinientos ochenta y nueve", 6)]

    an ARTÍCULO entity value
    [("419, 420 y 421", 5)]

    * A list of tuple where the first component is the index where the entity
    value starts and the second component a string representing the sentence.
    Each sentence must provide interpolation elements to insert values during
    tests. E.g.:

    [(5, De aplicación supletoria conforme {entity_nbor} {entity_value})]
    """
    return list(
        (entity_value, (start_index, start_index + entity_length, text))
        for (entity_value, entity_length) in entity_values
        for (start_index, text) in test_sentences
    )
