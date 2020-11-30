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
 python train.py train_model path_data_entrenamiento cantidad_de_iteraciones path_modelo lista_de_entidades_a_entrenar path_donde_se_guarda_el_mejor_modelo numero_flotante_de_losses_soportado
```

**Ejemplo**:

```bash
python train.py train_model Batch\ 02.json 2 modelo2020 [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD,NUM_TELEFÓNO,OCUPACIÓN/PROFESIÓN,PATENTE/DOMINIO,CORREO_ELECTRÓNICO,ARTÍCULO] el_mejor_modelo 24.7
```

#### Entrenar una serie de batches que estan en una carpeta

```bash
 python train.py all_files_in_folder Json\ 4/  2  modelo2020 [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD] el_mejor_modelo 100.0
```

#### Utilizar Displacy para probar entidades en un modelo

A tener en cuenta que se puede visualizar el Displacy en el puerto 5030 con cualquier browser
Ej: `localhost:5030`

```bash
 python train.py get_entities path_modelo texto_a_probar
```

**Ejemplo**:

```bash
 python train.py get_entities  modelo2020 "Soy un texto de prueba para detectar alguna entidad"
```

#### Utilizar Scorer para probar el modelo y obtener mas información

```bash
 python train.py scorer_model model_path text  annotations
```

**Ejemplo**:

```bash
 python train.py scorer_model modelo2020 "Carlos Alberto Mersu 99.999.999, Gabigol 23.213.456 y Tefi estaba en la hamaca con el dni 99999999" [(41,51,"NUM_DNI")]
```

#### Guardar en un archivo los logs del proceso:

```bash
 python train.py all_files_in_folder Json\ 4/  2  modelo2020 [FECHA,PER,DIRECCIÓN,NUM_DNI,NUM_CUIT_CUIL,EDAD,NACIONALIDAD] > nombre_de_archivo_de_logs.txt
```
