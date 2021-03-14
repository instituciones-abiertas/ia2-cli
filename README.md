# IA² | Línea de comandos

<p align="center">
  <a target="_blank" rel="noopener noreferrer">
    <img width="220px" src="public/images/ia2-logo.png" alt="IA²" />
  </a>
</p>
<br/>
<h4 align="center">Línea de comandos del proyecto IA²</h4>

## Stack Tecnológico

- Python, versión 3.7.6
- [Fire](https://github.com/google/python-fire)
- [Spacy](https://spacy.io/)

## Instalación

> Se recomienda instalar alguna herramienta para administrar versiones de python, como [pyenv](https://github.com/pyenv/pyenv) y alguna extensión para los ambientes virtuales, por ejemplo: [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Ambiente de desarrollo

***Inicializar la herramienta*** `precommit` ***para el control de sintaxis.***

```bash
pre-commit install
```

## Consideraciones

El proyecto no cuenta con datasets iniciales. Para construir nuestros dataset de entrenamiento y validación iniciales se utilizó la herramienta de etiquetado [Dataturks](http://dataturks.com/).

La línea de comandos contiene herramientas para transformar el etiquetado Dataturks a datasets soportados por Spacy. Para más información consulte el comando de ayuda de la línea de comandos.

## Circuito básico de prueba

El siguiente circuito de prueba contempla los siguientes procesos:

+ Descargar un modelo de spacy para utilizar como modelo base.
+ Creación de un modelo base
+ Agregar entidades al pipeline de reconocimiento de nombre de entidades del modelo base.
+ Ejecutar el entrenamiento, basándose en la configuración dada por el archivo de ejemplo `example_train_config.json`.

Descargar un modelo base

```bash
python -m spacy download es_core_news_lg
```

Crear un modelo basado en `es_core_news_lg` y guardarlo en la carpeta `models/base/2021-01-19`

```bash
python train.py create_custom_spacy_model \
  "es_core_news_lg" \
  "models/base/2021-01-19"
```

Agregar las entidades que nos interesan detectar en el modelo ([entidades soportadas](NER.md))

```bash
python train.py add_new_entity_to_model \
  "PER,LOC,DIRECCIÓN,OCUPACIÓN/PROFESIÓN,PATENTE_DOMINIO,ARTÍCULO" \
  "models/base/2021-01-19"
```

Entrenar un modelo (crear previamente un archivo `train_config.json` basado en `example_train_config.json`)

```bash
python train.py train example_tuning_hyperparams
```

## Línea de Comandos

### Ayuda

El flag `--help` proporciona información de los scripts disponibles.

```bash
python train.py --help
```

### Crear un modelo base

- `model_name`: nombre del modelo a utilizar como base del nuevo
- `output_path`: directorio donde se almacenará el nuevo modelo

```bash
python train.py create_custom_spacy_model <model_name> <output_path>
```

**Ejemplo para un modelo en español:**

```bash
python train.py create_custom_spacy_model \
  "es_core_news_lg" \
  "models/base/2021-01-19"
```

### Agregar entidades a un modelo

- `ents`: Strings de las entidades sin espacio
- `model_path`: directorio del modelo custom a utilizar

```bash
python train.py add_new_entity_to_model \
  <ents> \
  <model_path>
```

```bash
python train.py add_new_entity_to_model \
"PER,LOC,DIRECCIÓN" \
"models/base/2021-01-19"
```

### Entrenamiento de modelo

El entrenamiento guardará el mejor modelo (siempre que supere el threshold - leer parámetros de configuración), así como un archivo `history.csv` en la carpeta history (en la raiz del proyecto) en el que se explicitan parámetros y scores obtenidos por época (epoch).

- `config_name`: nombre de la configuración que se usará para entrenar el modelo, dicha debería estar en un archivo de configuración con el nombre `train_config.json`. 

```bash
python train.py train <config_name>
```

**Ejemplo:**

```bash
python train.py train example_tuning_hyperparams
```

El archivo de configuración `train_config.json` se debe generar a partir de `example_train_config.json`. Los parámetros disponibles para modificar son:

- `use_gpu`: valor booleano que determina si correr o no el entrenamiento usando el gpu. Se debe tener configurado CUDA toolkit y seguir este instructivo : [Ejecutar SpaCy con GPU](https://spacy.io/usage/#gpu).  **Nota**: tener en cuenta que el batch size afecta directamente el uso de memoria.
- `path_data_training`: directorio de la data para entrenar el modelo 
- `path_data_validation`: directorio de la data de validación para evaluar el modelo
- `path_data_testing`: directorio de la data de testing para evaluar el modelo. Si está incluida esta opción los conjuntos de entrenamiento y validación serán combinados y utilizados para entrenamiento. Ver [F. Chollet, _Deep Learning in Python_ , cap. 4.2](https://livebook.manning.com/book/deep-learning-with-python/chapter-4/44)
- `evaluate`: valor que determina que conjunto de datos usar para evaluar el modelo. Opciones `test` / `val`. `val` es el valor por defecto y no es necesario incluirlo
- `is_raw`: valor booleano que determina si el archivo será convertido (cuando is_raw sea True)
- `train_subset`:  si el valor es diferente de cero, se usará un subset del dataset tomado aleatoriamente (número entero)
- `model_path`: directorio del modelo custom a utilizar
- `save_model_path`: directorio donde se guardará el modelo generado a partir del entrenamiento
- `entities`: entidadas a ser usadas para el entrenamiento.
- `threshold`: valor a partir del cual se guardará un modelo, sólo si el score obtenido es mayor al threshold (número entero)
- `epochs`: cantidad de iteraciones / épocas en las que se entrenará el modelo (número entero)
- `optimizer`: aquí se pueden configurar los parámetros como el learning rate (tasa de aprendizaje) y otros presentes en el optimizador [Adam](https://thinc.ai/docs/api-optimizers#adam).
- `dropout`: porcentaje de _weights_ que se descartarán aleatoriamente para dar mayor variabilidad (número decimal) y evitar que el modelo memorice los datos de entrenamiento.
- `batch_size`: tamaño del batch (cantidad de textos) a utilizar para entrenar el modelo (número entero)
- `callbacks`: representa un objeto de arrays de callbacks a ser usados en el entrenamiento. Para ver dichas funciones ir al archivo `callbacks.py`.

### Reconocimiento con Displacy

El siguiente comando permite visualizar rapidamente resultados de un entrenamiento utilizando [displayCy](https://spacy.io/api/top-level#displacy). Disponibiliza un servidor en el puerto `5030`.

- `model_path`: directorio del modelo a utilizar para las pruebas
- `test_text`: string que represente un texto de prueba

```bash
python train.py display_text_prediction <model_path> <test_text>
```

**Ejemplo:**

```bash
python train.py display_text_prediction \
  models/base/2021-01-19 \
  "Soy un texto de prueba para detectar alguna entidad"
```

> Luego visitar `localhost:5030` desde un navegador.

### Conversion de datasets

El siguiente comando transforma una serie de documentos `.json` en formato dataturks a un dataset único, también en formato `.json`, soportado por la CLI de IA².

- `input_files_path`: directorio de archivos en formato dataturks
- `entities`: string que representa una lista de entidades, separadas por coma
- `output_file_path`: directorio de salida del dataset generado
- `num_files`: número de archivos que serán incluídos en la creación del dataset. Por defecto es `0` e incluye todos los archivos alojados en el directorio.

```bash
python train.py convert_dataturks_to_train_file \
  <input_files_path> \
  <entities> \
  <output_file_path> \
  <num_files>
```

**Ejemplo:**

Asume la existencia de un set de información etiquetada con Dataturks en el directorio data/raw/validation.

```bash
python train.py convert_dataturks_to_train_file \
  "data/raw/validation" \
  "PER, LOC, DIRECCIÓN" \
  "data/unified/validation.json"
```

### Correr comandos con timer

Ejecuta un comando en consola y guarda en el horario de comienzo y de fin en un log.

- `command_to_run`: Es el comando a ejecutar con parametros y espacios incluido.Va entre comillas dobles

```bash
python train.py run_command_with_timer <command>
```

**Ejemplo:**

```bash
python train.py run_command_with_timer "python train.py example_train_config example"
```

## Despliegue de modelo

El script `deploy_model.sh` se encarga de:

- Incluír **3 elementos al pipeline** (en orden de aparición y posterior al NER pipeline)
- Realizar modificaciones al código fuente del modelo: se asignan Language factories para cada componente.
- Generar un archivo `tar.gz` en el directorio `/dist`. Este bundle puede ser instalado mediante pip. Ejemplo: `pip install modelo.tar.gz`.

Pipelines que se incluyen:

- **EntityRuler**: `entity_ruler.py`
- **EntityMatcher**: `entity_matcher.py`
- **EntityCustom**: `entoty_custom.py`

Parámetros:

- `base_model`: directorio a un modelo base de origen, compatible con Spacy
- `model_name`: nombre del modelo a crear, sin espacios ni guiones bajos
- `version`: versión del modelo a crear
- `pipeline_components`: directorio a los módulos de pipeline components que se incluiran en el modelo

```bash
./deploy_model.sh <base_model> <model_name> <version> <pipeline_components>
```

**Ejemplo:**

```bash
./deploy_model.sh es_core_news_lg nombre-de-modelo 1.0 ./pipeline_components
```

## Tests

Algunos tests utilizan un modelo de spacy para realizar pruebas sobre texto plano. Por esta razón es necesario generar un archivo *`.env`*, utilizando *`.env.example`* como base. La variable `TEST_MODEL_PATH` del achivo `.env` debe contener la ruta hacia un modelo. Luego puede utilizar el siguiente comando para correr las pruebas:

```bash
make test
```

**Ejemplos:**

*`.env`*

```bash
# Un modelo entrenado y guardado dentro del directorio models/
export TEST_MODEL_FILE=models/path_to_my_model
# Otro ejemplo de un modelo de Spacy descargado utilizando (python -m spacy download es_core_news_lg)
export TEST_MODEL_FILE=es_core_news_lg
```

## Contribuciones

Por favor, asegúrese de leer los [lineamientos de contribución](CONTRIBUTING.md) antes de realizar Pull Requests.
