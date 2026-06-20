# overrides
data=EURLEX-4K
model=RetrieverBERT

text_max_length=256
label_max_length=32
label_enhancement=NONE          # raw label names only; needs no LLM/PMI artifacts
text_features_source=TXT
name=${label_enhancement}_${model}   # one consistent prefix across all stages

# sparse_retrieve  (BM25 query length is capped via data.max_query_words in the data config)
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[sparse_retrieve] \
    model=BM25 \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
done

# dense_retrieve fit
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fit] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=32 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fit_${name}_${data}_${fold_idx}.tmr
done

# dense_retrieve predict
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[predict] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/predict_${name}_${data}_${fold_idx}.tmr
done

# dense_retrieve eval
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[eval] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/eval_${name}_${data}_${fold_idx}.tmr
done

# fuse  (combines BM25_<DATA> with <name>_<DATA>)
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fuse] \
    model=$model \
    model.name=$name \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fuse_${name}_${data}_${fold_idx}.tmr
done

# aggregate  (merges head/tail of Fused_<name>_<DATA> and computes PS metrics)
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[aggregate] \
    model=$model \
    model.name=$name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_${name}_${data}_${fold_idx}.tmr
done
