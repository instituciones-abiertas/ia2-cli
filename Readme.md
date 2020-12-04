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

#### Helper

El flag `--help` proporciona información de los scripts disponibles.

```bash
python train.py --help
```

## Circuito basico de prueba

#### Descargar un modelo base

```bash
python -m spacy download es_core_news_lg
```

#### Crear un modelo en español vacío

- `model_name`: nombre del modelo a utilizar como base del nuevo
- `output_path`: directorio donde se almacenará el nuevo modelo

```bash
python train.py create_custom_spacy_model <model_name> <output_path>
```

**Ejemplo:**

```bash
python train.py create_custom_spacy_model \
  "es_core_news_lg" \
  "models/base/2020-12-01"
```

#### Entrenar con un batch

- `batch_path`: path de un archivo (`.json`) de entrenamiento.
- `iter_n`: número de iteraciones por batch.
- `model_path`: directorio del modelo custom a utilizar
- `training_entities`: lista de entidades a entrenar
- `output_path`: directorio donde se almacenará el modelo más óptimo
- `max_losses`: (número flotante) Máximo valor de Losses soportado

```bash
python train.py train_model \
  <batch_path> \
  <iter_n> \
  <model_path> \
  <training_entities> \
  <output_path> \
  <max_losses>
```

**Ejemplo:**

```bash
python train.py train_model \
  data/training/batches-2020-11-24/Batch_6.json \
  2 \
  models/base/2020-12-01 \
  "PER, LOC, DIRECCIÓN, OCUPACIÓN/PROFESIÓN, PATENTE/DOMINIO"
  models/best/2020-12-01 \
  24.7
```

#### Entrenar una serie de batches que estan en una carpeta

- `batches_path`: directorio de archivos (`.json`) de entrenamiento.
- `iter_n`: número de iteraciones por batch.
- `model_path`: directorio del modelo custom a utilizar
- `training_entities`: lista de entidades a entrenar
- `output_path`: directorio donde se almacenará el modelo más óptimo
- `max_losses`: (número flotante) Máximo valor de Losses soportado

```bash
python train.py train_model \
  <batches_path> \
  <iter_n> \
  <model_path> \
  <training_entities> \
  <output_path> \
  <max_losses>
```

**Ejemplo:**

```bash
python train.py all_files_in_folder \
  data/training/batches-2020-11-24/ \
  2 \
  models/base/2020-12-01 \
  "PER, LOC, DIRECCIÓN, OCUPACIÓN/PROFESIÓN, PATENTE/DOMINIO" \
  models/best/2020-12-01 \
  10.0
```

#### Utilizar Displacy para probar entidades en un modelo

El siguiente comando permite visualizar resultados de un entrenamiento utilizando [displayCy](https://spacy.io/api/top-level#displacy). Disponibiliza un servidor en el puerto `5030`.

- `model_path`: directorio del modelo a utilizar para las pruebas
- `test_text`: string que represente un texto de prueba

```bash
python train.py get_entities <model_path> <test_text>
```

**Ejemplo:**

```bash
python train.py get_entities \
  models/base/2020-12-01 \
  "Soy un texto de prueba para detectar alguna entidad"
```

> Luego visitar `localhost:5030` desde un navegador.

#### Utilizar Scorer para probar el modelo y obtener información sobre los resultados de pruebas

- `model_path`: directorio del modelo a utilizar para las pruebas
- `test_text`: string que represente un texto de prueba
- `annotations`: lista de ocurrencias de etiquetas `[(`)]`

```bash
python train.py scorer_model <model_path> <test_text> <annotations>
```

**Ejemplo**:

```bash
python train.py scorer_model \
  models/base/2020-12-01 \
  "Carlos Alberto Mersu 99.999.999, Gabigol 23.213.456 y Tefi estaba en la hamaca con el dni 99999999" \
  [(41,51,"NUM_DNI")]
```

#### Guardar en un archivo los logs del proceso

```bash
python train.py all_files_in_folder \
  data/training/batches-2020-11-24/ \
  2 \
  models/base/2020-12-01 \
  [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD] > logs_file_name.txt
```

#### Convertir un Batch Dataturks a JSON Spacy para usar CLI

```bash
convert_dataturks_to_training_cli
path_batch_a_convertir \
path_json_resultante \
lista_entidades_a_presevar \
```

##### Ejemplo

```bash
convert_dataturks_to_training_cli
Batch_05.json \
batch_05_spacy.json \
"PER, LOC" \
```
