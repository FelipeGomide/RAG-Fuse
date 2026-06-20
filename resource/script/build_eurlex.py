"""
Convert the raw EURLEX-4K extreme multi-label benchmark into the pkl format
consumed by the RAG-Fuse pipeline (see CLAUDE.md -> "Data format").

Raw inputs (already present in resource/dataset/EURLEX-4K/):
  - train_raw_texts.txt / test_raw_texts.txt : one (pre-processed) document per line
  - Y.trn.npz / Y.tst.npz                    : sparse (docs x labels) ground-truth matrices
  - label_map.txt                            : one label name per line, row i -> label id i

Outputs written into resource/dataset/EURLEX-4K/:
  - samples.pkl          : list[{idx, text_idx, text, labels_ids, labels}]
  - relevance_map.pkl    : {text_idx: [label_id, ...]}
  - label_cls.pkl        : {label_id: ["all", "head"|"tail"]}
  - text_cls.pkl         : {text_idx: ["head"|"tail", "all"]}
  - fold_0/{train,val,test}.pkl : lists of sample idx

Design notes
------------
* EURLEX-4K ships a canonical train/test split, so we build a single fold
  (fold_0) that respects it instead of random CV folds: test = official test
  documents; train/val = a 90/10 split of the official train documents.
  (val.pkl mainly augments the BM25 collection; fit reuses train.pkl for val.)
* head/tail follows the same rule the other datasets use: sort labels by
  document frequency (over all docs) descending and mark as "head" the smallest
  prefix whose cumulative frequency reaches >= 60% of total label occurrences;
  the rest are "tail". A document is "head" iff it carries at least one head
  label. This reproduces ACM/OHSUMED/TWITTER exactly.
"""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path("resource/dataset/EURLEX-4K")
HEAD_MASS = 0.60
VAL_FRACTION = 0.10
SEED = 42


def read_texts(path):
    # Documents are one-per-line; the file ends with a trailing newline.
    lines = Path(path).read_text(encoding="utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def read_label_names(path):
    names = Path(path).read_text(encoding="utf-8", errors="replace").split("\n")
    if names and names[-1] == "":
        names = names[:-1]
    return names


def doc_label_lists(Y):
    Y = Y.tocsr()
    return [sorted(int(c) for c in Y.indices[Y.indptr[r]:Y.indptr[r + 1]]) for r in range(Y.shape[0])]


def head_labels(freq, mass=HEAD_MASS):
    total = sum(freq.values())
    order = sorted(freq.items(), key=lambda kv: -kv[1])
    heads, cum = set(), 0
    for label_id, f in order:
        cum += f
        heads.add(label_id)
        if cum / total >= mass:
            break
    return heads


def main():
    train_texts = read_texts(DATA_DIR / "train_raw_texts.txt")
    test_texts = read_texts(DATA_DIR / "test_raw_texts.txt")
    label_names = read_label_names(DATA_DIR / "label_map.txt")

    Ytr = sp.load_npz(DATA_DIR / "Y.trn.npz")
    Yts = sp.load_npz(DATA_DIR / "Y.tst.npz")
    num_labels = Ytr.shape[1]

    assert len(train_texts) == Ytr.shape[0], (len(train_texts), Ytr.shape)
    assert len(test_texts) == Yts.shape[0], (len(test_texts), Yts.shape)
    assert len(label_names) == num_labels, (len(label_names), num_labels)
    assert Yts.shape[1] == num_labels

    train_labels = doc_label_lists(Ytr)
    test_labels = doc_label_lists(Yts)

    # ----- samples + relevance_map -------------------------------------------
    samples = []
    relevance_map = {}
    train_ids, test_ids = [], []

    def add_doc(text, labels_ids):
        idx = len(samples)              # list position == idx (pipeline indexes samples[idx])
        text_idx = idx                  # one unique document per sample
        samples.append({
            "idx": idx,
            "text_idx": text_idx,
            "text": text,
            "labels_ids": labels_ids,
            "labels": [label_names[i] for i in labels_ids],
        })
        relevance_map[text_idx] = labels_ids
        return idx

    for text, labels in zip(train_texts, train_labels):
        train_ids.append(add_doc(text, labels))
    for text, labels in zip(test_texts, test_labels):
        test_ids.append(add_doc(text, labels))

    assert all(s["idx"] == i for i, s in enumerate(samples))
    assert all(len(l) > 0 for l in relevance_map.values()), "found a doc with no labels"

    # ----- head/tail classes --------------------------------------------------
    freq = Counter()
    for labels in relevance_map.values():
        freq.update(labels)
    for lid in range(num_labels):       # ensure every label id has a frequency entry
        freq.setdefault(lid, 0)
    heads = head_labels(freq)

    label_cls = {lid: ["all", "head" if lid in heads else "tail"] for lid in range(num_labels)}
    text_cls = {}
    for text_idx, labels in relevance_map.items():
        is_head = any(l in heads for l in labels)
        text_cls[text_idx] = ["head" if is_head else "tail", "all"]

    # ----- fold_0 (official split; val carved from train) --------------------
    rng = np.random.default_rng(SEED)
    perm = np.array(train_ids)
    rng.shuffle(perm)
    n_val = int(round(VAL_FRACTION * len(perm)))
    val_split = sorted(int(i) for i in perm[:n_val])
    train_split = sorted(int(i) for i in perm[n_val:])
    test_split = sorted(test_ids)

    fold_dir = DATA_DIR / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    def dump(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    dump(samples, DATA_DIR / "samples.pkl")
    dump(relevance_map, DATA_DIR / "relevance_map.pkl")
    dump(label_cls, DATA_DIR / "label_cls.pkl")
    dump(text_cls, DATA_DIR / "text_cls.pkl")
    dump(train_split, fold_dir / "train.pkl")
    dump(val_split, fold_dir / "val.pkl")
    dump(test_split, fold_dir / "test.pkl")

    # ----- report ------------------------------------------------------------
    n_head = sum(1 for c in label_cls.values() if "head" in c)
    head_text = sum(1 for c in text_cls.values() if "head" in c)
    print(f"samples           : {len(samples)} (train {len(train_texts)} + test {len(test_texts)})")
    print(f"labels            : {num_labels}  head {n_head} / tail {num_labels - n_head}")
    print(f"avg labels/doc    : {sum(len(l) for l in relevance_map.values()) / len(relevance_map):.2f}")
    print(f"text head/tail    : {head_text} head / {len(text_cls) - head_text} tail")
    print(f"fold_0 splits     : train {len(train_split)}  val {len(val_split)}  test {len(test_split)}")
    print("wrote: samples.pkl, relevance_map.pkl, label_cls.pkl, text_cls.pkl, fold_0/{train,val,test}.pkl")


if __name__ == "__main__":
    main()
