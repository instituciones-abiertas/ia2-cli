# Reconocimiento de Entidades Nombradas (NER)

## Modelos Estadístico

El componente NER del modelo de IA² fue entrenado para detectar las siguientes entidades.

| Entidad   | Descripción | Ejemplo  |
| :-------: | :---------: | :------: |
|PER|Persona|Para dictar sentencia en la presente causa nº 3121/27, caratulada “Contra **Luis Gonzalez** por 1189 bis (2) – tenencia de arma de fuego de uso” |
|LOC|Locación, Ciudad, etc|el inmueble sito en la calle Olavarría 2120, piso 2, de la **CABA**|
|DIRECCIÓN|Domicilio particular|el inmueble sito en la calle **Olavarría 2120**, piso 2, de la CABA |
|ARTÍCULO|Número de artículo en referencia a leyes|Copia de la notificación del **art. 36** de la Convención de Viena|

## Patrones

A la información detectada mediante el aprendizaje automático del modelo, se agregan reglas para la detección de más entidades mediante expresiones regulares y listados específicos detallados (por ejemplo: Bancos de la República Argentina).

| Entidad   | Descripción | Ejemplo  | Referencia | Pipeline |
| :-------: | :---------: | :------: | :--------: | :------: |
|BANCO|bancos de la república argentina|la imputada fue al **Banco Credicoop Cooperativo Limitado** a hacer un trámite| [Bancos soportados](pipeline_components/entity_ruler.py#L61)| Ruler |
|CBU|clave bancaria uniforme (22 caracteres numéricos)|la cuenta con CBU **0140323501111111500292** del Banco Credicoop Cooperativo Limitado| 22 digitos seguidos| Ruler |
|CORREO_ELECTRONICO|correos electrónicos|el imputado envió dicho texto desde el correo electrónico **luis_gonzales@mail.com**|[Formato](https://github.com/explosion/spaCy/blob/047fb9f8b8cfe99abc8455aa990fa2c2dd3d4c84/spacy/lang/lex_attrs.py#L10) | Ruler |
|EDAD|Edad de alguna persona|el imputado tiene **21** años de edad| Número seguidos de *años* y que contiene *edad* en oración| Custom |
|ESTUDIOS|trayectoria de estudios de las personas (niveles, completo / incompleto)|el imputado posee **estudios secundarios completos**|[Combinaciones soportadas](pipeline_components/entity_ruler.py#L350)| Ruler |
|FECHA|fecha escrita|Que el día **3 de julio** del corriente año|[Combinaciones soportadas](pipeline_components/entity_ruler.py#L399)| Ruler |
|FECHA_HECHO|fecha de cuando sucedió el hecho|Solo identificable por intervención de personas| | Etiquetado Manual |
|FECHA_NÚMERICA|fecha en números|Que el **03/07/2021** la situación sea aclarada|[Formatos soportados](pipeline_components/entity_ruler.py#L380)| Ruler |
|FECHA_RESOLUCION|fecha escrita|Buenos Aires, **3 de julio del 2020**|Primera fecha que aparece (primeros tokens, o entre las 1eras entidades si antes hay loc o cuij)| Custom |
|FISCAL|nombre de fiscal nombrado en la causa|Fiscal: **Maria Gonzalez**|*Fiscal* o *fiscalía* antes de identificación como *PER* | Custom |
|JUEZX|nombre de juez/jueza nombradx en la causa|Juez: **Luis Gonzalez**| *Juez* antes de identificación como *PER*| Custom |
|DEFENSORX|nombre de defensor/a nombrado en la causa|Defensora: **Maria Gonzalez**|*Defensor* o *Defensora* antes de  identificación como *PER* | Custom |
|LEY|ley citada en la resolución|otros elementos presuntamente constitutivos de una infracción a la ley **23737**| Palabra *ley* antes de número| Custom |
|LINK|enlace web|ingresó al sitio **https://www.google.com/** y buscó las palabras “como armar una bomba”|[Formato soportado](https://github.com/explosion/spaCy/blob/047fb9f8b8cfe99abc8455aa990fa2c2dd3d4c84/spacy/lang/lex_attrs.py#L124)| Ruler |
|LUGAR_HECHO|lugar donde sucedió el evento|Solo identificable por intervención de personas| | Etiquetado Manual |
|MARCA_AUTOMÓVIL|marcas de automóviles|el imputado manejaba un vehículo marca **volkswagen** con patente FHG-456| [Marcas soportadas](pipeline_components/entity_ruler.py#L1)| Ruler |
|NACIONALIDAD|nacionalidad de una persona|la imputada de nacionalidad argentina|[Soportadas](pipeline_components/entity_ruler.py#L487)| Ruler |
|NOMBRE_ARCHIVO|nombres de archivos|envió el archivo **carta_documento.docx** mediante el correo eletrónico luis_gonzalez@mail.com|[Tipos soportados, incluye extensión](pipeline_components/entity_ruler.py#L594) | Ruler |
|NUM|Números genéricos|Cualquier otra entidad númerica| Cualquier otro número que no haga referencia a páginas, articulos y unidades de medida| Entity Matcher |
|NUM_ACTUACIÓN|número de actuación|Actuación Nro: **15221125/2020**|Palabra vecina tiene *nro actuación* | Custom |
|NUM_CAUSA|número de identificación de la causa|Para dictar sentencia en la presente causa nº **3121/27**, caratulada “Contra Luis Gonzalez por 1189 bis (2) – tenencia de arma de fuego de uso”| Cuando encuentra *caso* o *n° causa* previamente | Custom |
|NUM_CUIJ|número de CUIJ|CUIJ: **C-01-00480932-3**|Palabra vecina es CUIJ| Ruler, Custom |
|NUM_CUIT_CUIL|número de CUIT / CUIL|Luis Gonzalez con CUIT **20-32481145-7**| [Formato soportado](pipeline_components/entity_ruler.py#L593)| Ruler |
|NUM_DNI|número de documento nacional de identidad|Luis Gonzalez, DNI **32.481.145**|[Formatos soportados](pipeline_components/entity_ruler.py#L361)| Ruler |
|NUM_EXPEDIENTE|número de expediente|15) Copias del expediente **9123/2011**;| Palabras vecinas tiene *n°* y/o *expediente*| Cusdtom |
|NUM_IP|direcciones IP|desde un dispositivo con dirección IP **200.111.111.111**|[Formatos soportados](pipeline_components/entity_ruler.py#L374)| Ruler, Custom |
|NUM_TELÉFONO|números de teléfono|el llamado se realizó desde el **4123-4123**|[Formatos soportados](pipeline_components/entity_ruler.py#L367)| Ruler, Custom |
|PASAPORTE|número de pasaporte argentino|Maria Gonzalez, pasaporte nro **ABC123456** | [Formato soportado](pipeline_components/entity_ruler.py#L596)| Ruler |
|PATENTE_DOMINIO|patente de vehículo|el imputado manejaba un vehículo marca volkswagen con patente **FHG-456**|[Formatos soportados](pipeline_components/entity_ruler.py#L330)| Ruler |
|PERIODO|período de tiempo|con prisión de **seis años**, si se tratare de un instrumento privado|[Formatos soportados](pipeline_components/entity_custom.py#L5)| Custom |
|SECRETARIX|nombre de secretario/secretaria nombradx en la causa|Secretaria: **María Gonzalez**| *Secretario* antes de identificación como *PER* | Custom |
|USUARIX|del usuario "juan21" de red social|Usuarios de web o redes sociales|Soporta patrón *del usuario "nombredelusuario"* | Ruler |
