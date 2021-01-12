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
***Inicializar precomitt en el repo***
Es necesario instalar  precommit en el repo:
```bash
pre-commit install
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

#### Crear un modelo vacío

- `model_name`: nombre del modelo a utilizar como base del nuevo
- `output_path`: directorio donde se almacenará el nuevo modelo

```bash
python train.py create_custom_spacy_model <model_name> <output_path>
```

**Ejemplo para un modelo en español:**

```bash
python train.py create_custom_spacy_model \
  "es_core_news_lg" \
  "models/base/2020-12-01"
```

#### Agregar entidades a un modelo

- `ents`: Strings de las entidades sin espacio
- `model_path`: directorio del modelo custom a utilizar

```bash
python train.py add_new_entity_to_model \
  <ents> \
  <model_path> \
```

```bash
python train.py add_new_entity_to_model \
"PER,LOC,DIRECCIÓN,OCUPACIÓN/PROFESIÓN,PATENTE_DOMINIO,ARTÍCULO" \
modelos/modelo10-12 \
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
python train.py train_all_files_in_folder \
  <batches_path> \
  <iter_n> \
  <model_path> \
  <training_entities> \
  <output_path> \
  <max_losses>
```

**Ejemplo:**

```bash
python train.py train_all_files_in_folder \
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
python train.py display_text_prediction <model_path> <test_text>
```

**Ejemplo:**

```bash
python train.py display_text_prediction \
  models/base/2020-12-01 \
  "Soy un texto de prueba para detectar alguna entidad"
```

> Luego visitar `localhost:5030` desde un navegador.

#### Utilizar Scorer para probar el modelo y obtener información sobre los resultados de pruebas

- `model_path`: directorio del modelo a utilizar para las pruebas
- `test_text`: string que represente un texto de prueba
- `annotations`: lista de ocurrencias de etiquetas `[(`)]`

```bash
python train.py evaluate <model_path> <test_text> <annotations>
```

**Ejemplo:**

```bash
python train.py scorer_model \
  models/base/2020-12-01 \
  "Carlos Alberto Mersu 99.999.999, Gabigol 23.213.456 y Tefi estaba en la hamaca con el dni 99999999" \
  [(41,51,"NUM_DNI")]
```

#### Guardar en un archivo los logs del proceso

```bash
python train.py train_all_files_in_folder \
  data/training/batches-2020-11-24/ \
  2 \
  models/base/2020-12-01 \
  "FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD" > logs_file_name.txt
```

#### Conversion de datasets

El siguiente comando transforma una serie de documentos `.json` en formato dataturks a un dataset único, también en formato `.json`, soportado por la CLI de Spacy.

```bash
python train.py convert_dataturks_to_training_cli \
  <input_files_path> \
  <entities> \
  <output_file_path>
```

**Ejemplo:**

```bash
python train.py convert_dataturks_to_training_cli \
  "data/raw/training" \
  "PER, LOC, DIRECCIÓN, OCUPACIÓN/PROFESIÓN, ARTÍCULO, PATENTE_DOMINIO" \
  "data/spacy/training/training_data.json"
```

#### Correr un comando con un timer:

Poder correr un comando en consola y se guarda en el logger el horario de comienzo y de fin

- `command_to_run`: Es el comando a ejecutar con parametros y espacios incluido.Va entre comillas dobles

```bash
python train.py run_command_with_timer "command_to_run"
```

```bash
 python train.py run_command_with_timer "python -m spacy train \
   es \
   modelos/09-12-2020-03 \
   DatasetIntegrados/entrenamiento_circuito3.json \
   DatasetIntegrados/validacion_circuito1.json  \
   --pipeline=ner \
   --n-iter=2 \
   --base-model modelos/modelo10-12"
```
