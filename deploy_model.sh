#!/bin/bash

if [ $# -ne 4 ]; then
    echo "No enviaste los parametros necesarios"
    echo Ejemplo: "$0 MODELO_ORIGEN MODELO_NOMBRE MODELO_VERSION MODELO_COMPONENTES"
    exit 1
fi

MODELO_ORIGEN=$1
MODELO_NOMBRE=$2
MODELO_VERSION=$3
MODELO_COMPONENTES=$4
MODELO_COMPLETO="es_$MODELO_NOMBRE-$MODELO_VERSION"

DIR_DIST="dist"
DIR_MODELO_DEPLOY="$DIR_DIST/modelo_deploy"
DIR=`pwd`
rm -rf $DIR_DIST/
mkdir -p $DIR_MODELO_DEPLOY
python ./train.py build_model_package $MODELO_ORIGEN $DIR_MODELO_DEPLOY $MODELO_NOMBRE $MODELO_VERSION $MODELO_COMPONENTES
cd $DIR_MODELO_DEPLOY/$MODELO_COMPLETO
python setup.py sdist
cp $DIR_DIST/* ../../
cd $DIR
rm -rf $DIR_MODELO_DEPLOY
