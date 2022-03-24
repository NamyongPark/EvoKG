#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage) $0 [seed]"
  exit 0
fi
SEED="$1"

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
cd "${scriptDir}"/../../src/ || exit

python3 train.py \
--graph ICEWS_500 \
--seed "${SEED}" \
--log-dir evokg_timepred_seed"${SEED}" \
--clean-up-run-best-checkpoint \
--epochs 1000 \
--combiner-gconv none \
--static-dynamic-combine-mode concat \
--num-rnn-layers 1  \
--static-entity-embed-dim 200 \
--temporal-dynamic-entity-embed-dim 200 \
--structural-dynamic-entity-embed-dim 200 \
--dropout 0.2 \
--early-stop \
--early-stop-criterion MAE \
--patience 5 \
--eval-every 1 \
--embedding-updater-structural-gconv NONE \
--lr 0.0005 \
--gpu 0 \
--optimize time --eval time \
--time-pred-eval \
--rnn-truncate-every 20 \
--inter-event-time-mode min_inter_event_times "${@:2}"