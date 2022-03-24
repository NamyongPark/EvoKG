#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage) $0 [seed]"
  exit 0
fi
SEED="$1"

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
cd "${scriptDir}"/../../src/ || exit


python3 train.py \
--graph GDELT \
--seed "${SEED}" \
--log-dir evokg_linkpred_seed"${SEED}" \
--result-file-prefix opt_edge_ \
--clean-up-run-best-checkpoint \
--dropout 0.0 \
--eval-from 0 \
--eval-every 3 \
--early-stop \
--patience 4 \
--lr 0.001 \
--gpu 0 \
--optimize edge \
--eval edge \
--rnn-truncate-every 40 \
--full-link-pred-validation \
--full-link-pred-test "${@:2}"


python3 train.py \
--graph GDELT \
--seed "${SEED}" \
--log-dir evokg_linkpred_seed"${SEED}" \
--result-file-prefix opt_both_ \
--clean-up-run-best-checkpoint \
--dropout 0.0 \
--eval-from 0 \
--eval-every 1 \
--early-stop \
--patience 5 \
--lr 0.0005 \
--gpu 0 \
--load-best edge \
--optimize both \
--eval both \
--rnn-truncate-every 40 \
--full-link-pred-validation \
--full-link-pred-test \
--inter-event-time-mode node2node_inter_event_times "${@:2}"