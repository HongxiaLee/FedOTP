#!/bin/bash

cd ...

# custom config
DATA="DATA/"
MODEL=FedOTP
TRAINER=GLP_OT
OT=COT
TOP_PERCENT=0.80
EPS=0.01
THRESH=0.001
MAX_ITER=100
PRETRAINED=True
LR=0.001
GAMMA=1
USERS=10
FRAC=1
ROUND=10
NUM_PROMPT=2
#DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
IID=False
CSC=False  # class-specific context (False or True)
USEALL=False
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in caltech101 oxford_flowers oxford_pets dtd
do
  for SHOTS in 1 2 4 8 16
  do
    for SEED in 0 1
    do
      DIR=output/${DATASET}/${MODEL}_${TRAINER}_new${OT}_toppercent${TOP_PERCENT}_eps${EPS}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
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
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --num_prompt ${NUM_PROMPT} \
        --num_shots ${SHOTS} \
        --train_batch_size ${SHOTS} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/rn50.yaml \
        --output-dir ${DIR} 
      fi
    done
  done
done

