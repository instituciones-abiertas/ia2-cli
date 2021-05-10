# Guía de contribución de IA²-Training-Cli

¡Hola! Estamos muy emocionadxs de que estes interesadx en contribuir a IA2-Training-Cli. Antes de enviar Pull Requests, asegurate de tomarte un momento y leer las siguientes pautas:

+ [Código de conducta](https://github.com/instituciones-abiertas/ia2-cli/blob/master/CODE_OF_CONDUCT.md)
+ [Pautas para la apertura de issues](#pautas-para-la-apertura-de-issues)
+ [Pautas para la apertura de Pull Requests](#pautas-para-la-apertura-de-pull-requests)
+ [Configuración de desarrollo](#configuración-de-desarrollo)
+ [Estructura del proyecto](#estructura-del-proyecto)

## Pautas para la apertura de issues

- Utiliza siempre nuestros templates de issue para [**bugs**](url-del-template) o para [**features**](url-del-template) para crear issues.

## Pautas para la apertura de Pull Requests

+ La rama `master` es solo un snapshot de la última versión estable. Todo el desarrollo debe realizarse en ramas dedicadas que apunten a la rama `develop`. **No envíes PRs contra la rama `master`.**

+ Siempre realizar un checkout partiendo de la rama en cuestión, ej: `develop` y realizar el merge contra esa misma rama al finalizar. Siga esta convención para la nueva rama: `númeroDeIssue-usuarioDeGithub-títuloDeCommit`.

+ Esta bien realizar varios commit pequeños mientras trabajas en el PR. Podemos realizar un squash antes de mergear la rama, si es necesario.

+ Si agregas una nueva característica:
  + Agrega un caso de prueba
  + Proporciona una razón convincente para agregar esta función. Idealmente, primero debes abrir un issue comentando la sugerencia y aguardar que se apruebe antes de trabajar en él.

+ Si arreglas un bug:
  + Si estas resolviendo un caso especial sigue la convención de nomenclatura de ramas mencionada anteriormente.
  + Proporciona una descripción detallada de la resolución del bug en el PR. Se prefiere una demostración en vivo.

## Configuración de desarrollo

Necesitarás [**Python**](https://www.python.org) **versión 3.7.6 o posterior**.

Después de clonar el repositorio forkeado, sigue las instrucciones de desarrollo en [README.md](README.md#Instalación)

### Escritura de commits

No esperamos ninguna convención estricta, pero agradeceríamos que resumieras de qué trata el contenido de tus modificaciones al escribir un commit.

## Estructura del proyecto

+ **`callbacks.py`**: módulo python donde se define la implementación de las funciones callback. Estas funciones se declaran en el archivo de configuración de entrenamiento bajo el atributo `callbacks`.

+ **`example_train_config.json`**: archivo de configuración de entrenamiento de ejemplo.

+ **`train.py`**: entrypoint principal de scripts para la línea de comandos de entrenamiento.

+ **`history`**: `<DIR>` contiene el historial con los resultados de entrenamiento de un modelo dado.

+ **`models`**: `<DIR>` directorio utilizado para contener modelos base a utilizar durante el entrenamiento. Se puede configurar desde el archivo de configuración de entrenamiento.

+ **`logs`**: `<DIR>` directorio dedicado al alojamiento de archivos de log que producen los scripts.
