# Dataset sources

The built dataset under `resource/dataset/` is **not** versioned (it is git-ignored
and contains files over GitHub's 100 MB limit). Instead, this directory holds a slim
source archive from which the full runtime dataset is regenerated.

## EURLEX-4K

`EURLEX-4K-source.tar.gz` (~39 MB) contains only the raw inputs `build_eurlex.py`
needs: `train_raw_texts.txt`, `test_raw_texts.txt`, `label_map.txt`, `Y.trn.npz`,
`Y.tst.npz`. The unused XLNet/sparse feature files (`X.*`) and the original 150 MB
`Eurlex-4K.tar.gz` are intentionally excluded.

### Rebuild

```bash
mkdir -p resource/dataset/EURLEX-4K
tar xzf resource/dataset_src/EURLEX-4K-source.tar.gz -C resource/dataset/EURLEX-4K
python resource/script/build_eurlex.py
```

This regenerates, byte-for-byte (SEED=42), the files the pipeline consumes:
`samples.pkl`, `relevance_map.pkl`, `label_cls.pkl`, `text_cls.pkl`, and
`fold_0/{train,val,test}.pkl`.
