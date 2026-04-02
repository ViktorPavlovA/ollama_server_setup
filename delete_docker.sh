#!/bin/bash

docker compose --env-file ./docker_model/.env_model \
               --env-file ./docker_vdb/.env_vdb \
               down -v --rmi all