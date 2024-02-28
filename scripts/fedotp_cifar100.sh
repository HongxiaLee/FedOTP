#!/bin/bash

cd ...

# custom config
DATA="DATA/"
MODEL=FedOTP
TRAINER=GLP_OT
PRETRAINED=True
OT=COT
TOP_PERCENT=0.8
EPS=0.1
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=1
USERS=100
FRAC=0.1
ROUND=150
NUM_PROMPT=2
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CTXINIT=False
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
BETA=0.3
# SEED=1
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in cifar100
do
  for PARTITION in noniid-labeldir100
  do
    for SEED in 1
    do
      DIR=output/${DATASET}_${PARTITION}_beta${BETA}/${MODEL}_${TRAINER}_${OT}_${TOP_PERCENT}_eps${EPS}/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python federated_main.py \
        --root ${DATA} \
        --model ${MODEL} \
        --seed ${SEED} \
        --num_users ${USERS} \
        --frac ${FRAC} \
        --lr ${LR} \
        --OT ${OT} \
        --top_percent ${TOP_PERCENT} \
        --eps ${EPS} \
        --thresh ${THRESH} \
        --max_iter ${MAX_ITER} \
        --gamma ${GAMMA} \
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --n_ctx ${NCTX} \
        --num_prompt ${NUM_PROMPT} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/vit_b16.yaml \
        --output-dir ${DIR}
      fi
    done
  done
done
