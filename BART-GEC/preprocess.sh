fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "data/bpe/train.bpe" \
  --validpref "data/bpe/valid.bpe" \
  --destdir "gec_data-bin/" \
  --workers 10 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
