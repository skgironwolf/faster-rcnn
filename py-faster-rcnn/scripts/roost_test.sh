#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or cnn_data.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=1000
    ;;
  cnn_data)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="CNNDATA_train"
    TEST_IMDB="CNNDATA_test" #CNNDATA_trainval
    PT_DIR="pascal_voc"
    ITERS=0
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ../../tools/train_net.py --gpu ${GPU_ID} \
  --solver ../../models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ../../data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg ../../experiments/cfgs/roost.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
echo ${NET_FINAL}

time ../../tools/test_net.py --gpu ${GPU_ID} \
  --def ../../models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ../../output/default/voc_2007_trainval/zf_faster_rcnn_iter_140000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg ../../experiments/cfgs/roost.yml \
  ${EXTRA_ARGS}
