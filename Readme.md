## Util para entrenar Spacy

#### Stack

- Phython
- [Fire](https://github.com/google/python-fire)
- [Spacy](https://spacy.io/)

#### Pre-requisitos:

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

#### Uso:

Se puede pedir ayuda para saber los parametros:

```bash
python train.py --help
```

## Circuito basico de prueba

#### Crear un modelo en español vacio

```bash
python train.py create_custom_spacy_model "es_core_news_sm" "path_donde_se_quiere_guardar_el_modelo"
```

**Ejemplo**:

```bash
python train.py create_custom_spacy_model "es_core_news_sm" "./modelo2020"
```

#### Entrenar con un batch

```bash
 python train.py train_model path_data_entrenamiento cantidad_de_iteraciones path_modelo lista_de_entidades_a_entrenar
```

**Ejemplo**:

```bash
python train.py train_model Batch\ 02.json 2 modelo2020 [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD,NUM_TELEFÓNO,OCUPACIÓN/PROFESIÓN,PATENTE/DOMINIO,CORREO_ELECTRÓNICO,ARTÍCULO]
```

#### Entrenar una serie de batches que estan en una carpeta

```bash
 python train.py all_files_in_folder Json\ 4/  2  modelo2020 [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD]
```
