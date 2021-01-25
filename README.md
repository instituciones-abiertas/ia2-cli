## Util para entrenar Spacy

#### Stack

- Python
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

#### Entrenar un modelo
Dicho entrenamiento guardará el mejor modelo (siempre que supere el threshold - leer parámetros de configuración), así como un archivo `history.csv` en la carpeta history (en la raiz del proyecto) en el que se explicitan parámetros y scores obtenidos por época (epoch).

- `config_name`: nombre de la configuración que se usará para entrenar el modelo, dicha debería estar en un archivo de configuración con el nombre `train_config.json`. 

```bash
python train.py train <config_name>
```

**Ejemplo:**

```bash
python train.py train train_config
```

El archivo de configuración `train_config.json` se debe generar a partir de `example_train_config.json`. Los parámetros disponibles para modificar son:
- `path_data_training`: directorio de la data para entrenar el modelo 
- `path_data_validation`: directorio de la data para entrenar el modelo
- `is_raw`: valor booleano que determina si el archivo será convertido (cuando is_raw sea True)
- `train_subset`:  si el valor es diferente de cero, se usará un subset del dataset (número entero)
- `model_path`: directorio del modelo custom a utilizar
- `save_model_path`: directorio donde se guardará el modelo generado a partir del entrenamiento
- `entities`: entidadas a ser usadas para el entrenamiento.
- `threshold`: valor a partir del cual se guardará un modelo, sólo si el score obtenido es mayor al threshold (número entero)
- `epochs`: cantidad de iteraciones / épocas en las que se entrenará el modelo (número entero)
- `optimizer`: aquí se pueden configurar los parámetros learning rate (tasa de aprendizaje) y beta1 del Adam Solver.
- `dropout`: porcentaje de data de entrenamiento que se descartará aleatoriamente para dar mayor variabilidad (número decimal)
- `batch_size`: tamaño del batch a utilizar para entrenar el modelo (número entero)
- `callbacks`: representa un objeto de arrays de callbacks a ser usados en el entrenamiento. Para ver dichas funciones ir al archivo `callbacks.py`.


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

#### Conversion de datasets

El siguiente comando transforma una serie de documentos `.json` en formato dataturks a un dataset único, también en formato `.json`, soportado por la CLI de Spacy.

```bash
python train.py convert_dataturks_to_train_file \
  <input_files_path> \
  <entities> \
  <output_file_path>
```

**Ejemplo:**

```bash
python train.py convert_dataturks_to_train_file \
  "data/raw/training" \
  "PER, LOC, DIRECCIÓN, OCUPACIÓN/PROFESIÓN, ARTÍCULO, PATENTE_DOMINIO" \
  "data/spacy/training/training_data.json"
```

#### Correr un comando con un timer:

Poder correr un comando en consola y se guarda en el logger el horario de comienzo y de fin

- `command_to_run`: Es el comando a ejecutar con parametros y espacios incluido.Va entre comillas dobles

```bash
python utils.py run_command_with_timer "command_to_run"
```

```bash
 python utils.py run_command_with_timer "python -m spacy train \
   es \
   modelos/09-12-2020-03 \
   DatasetIntegrados/entrenamiento_circuito3.json \
   DatasetIntegrados/validacion_circuito1.json  \
   --pipeline=ner \
   --n-iter=2 \
   --base-model modelos/modelo10-12"
```

#### Desplegar modelo

Este script **deploy_model.sh** se encarga de:
- Agregar **3 elementos al pipeline** (en orden y luego del NER)
- Agregar modificaciones al código fuente del modelo agregando Language factories para cada componente
- Generar un archivo tar.gz dentro de **/dist** (para luego instalar mediante `pip install modelo`)

Pipelines agregados:
- EntityRuler (entity_ruler.py)
- EntityMatcher (entity_matcher.py)
- EntityCustom (entoty_custom.py)

Ejemplo de uso:
```
./deploy_model.sh es_core_news_lg juzgado10 1.0 ./pipeline_components
```

Parametros en orden:
- MODELO_ORIGEN: nombre o directorio del modelo de origen
- MODELO_NOMBRE: nombre del nuevo modelo a crear
- MODELO_VERSION: versión del modelo a crear
- MODELO_COMPONENTES: directorio donde estan los archivos de componentes (entity_ruler.py, entity_matcher.py, entity_custom.py)
