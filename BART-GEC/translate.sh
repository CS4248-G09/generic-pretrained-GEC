OUT=output
model=./bart.large

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python translate.py \
  $model \
  data/test/ABCN.test.bea19.orig \
  $OUT \