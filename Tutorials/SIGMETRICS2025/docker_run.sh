#!/bin/sh

PROJ=riscv_cross_compiler
CONTAINER_NAME=${PROJ}_container

docker exec -it --user root ${CONTAINER_NAME} bash
