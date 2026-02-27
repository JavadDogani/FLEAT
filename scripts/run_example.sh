#!/usr/bin/env bash

python -m fleat_project.main \
  --method fleat \
  --model googlenet \
  --dataset mnist \
  --num_clients 10 \
  --rounds 20 \
  --partition dirichlet \
  --dirichlet_alpha 0.3 \
  --device cuda \
  --server_device cuda
