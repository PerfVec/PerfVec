#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

PROJ="riscv_cross_compiler"
DOCKERFILE_PATH="${GITTOP}/docker"

IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container

# output dir mounting
DST_DIR="${GITTOP}/Tutorials/SIGMETRICS2025"
DST_TARGET_DIR="/home"

# build dockerfile to generate docker image
echo "[${PROJ}] - pull docker image from dockerhub..."

docker pull khsubnl/riscv_cross_compiler

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

echo "[${PROJ}] - build docker container..."

docker run -d \
         -it \
         --name ${CONTAINER_NAME} \
         -u $(id -u $USER):$(id -g $USER) \
         --mount type=bind,source=${DST_DIR},target=${DST_TARGET_DIR} \
         ${IMAGE_NAME} \
         bash
         
