import heapq
import logging
import pickle
from pathlib import Path

import pandas as pd
from ranx import evaluate, Run, Qrels
from retriv import SparseRetriever

from source.helper.Helper import Helper


class SparseRetrieverHelper(Helper):

    def __init__(self, params):
        super(SparseRetrieverHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.relevance_map = self._load_relevance_map()
        self.label_cls = self._load_labels_cls()
        self.text_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        logging.basicConfig(level=logging.INFO)

    def _get_features(self, sample):
        if self.params.data.text_features_source == "TXT":
            return sample["text"]
        elif self.params.data.text_features_source == "KWD":
            return " ".join([kwd[0] for kwd in sample["keywords"]])
        else:
            raise Exception("Features must be TXT or KWD.")

    def _get_query_features(self, sample):
        # Bound the query length (in words) when data.max_query_words is set.
        # Long-document corpora (e.g. EURLEX-4K: avg ~1.3k, max ~28k words/doc) make
        # retriv's batch scoring allocate huge per-query posting gathers and OOM.
        # Truncating only the query keeps peak RAM flat; the collection stays full.
        text = self._get_features(sample)
        max_words = self.params.data.get("max_query_words", None)
        if max_words:
            text = " ".join(text.split()[:max_words])
        return text

    def _get_collection_and_queries(self, samples, fold_idx, split):
        logging.info(f"Getting collection and queries for {self.params.data.name} on fold {fold_idx} and split {split}")
        collection, queries = [], []
        if split == "train" or split == "val":
            for idx in self._load_split_ids(fold_idx, split=split):
                collection.append({
                    "id": f"doc_{idx}",
                    "text": self._get_features(samples[idx])
                })
                queries.append({
                    "id": f"q_{idx}",
                    "text": self._get_query_features(samples[idx])
                })
        elif split == "test":
            for idx in list(set(self._load_split_ids(fold_idx, split="train")) | set(
                    self._load_split_ids(fold_idx, split="val"))):
                collection.append({
                    "id": f"doc_{idx}",
                    "text": self._get_features(samples[idx])
                })
            for idx in self._load_split_ids(fold_idx, split="test"):
                queries.append({
                    "id": f"q_{idx}",
                    "text": self._get_query_features(samples[idx])
                })

        return collection, queries

    def _get_sparse_retriever(self, collection, index_name, model):
        SparseRetriever.delete(index_name)
        sr = SparseRetriever(
            index_name=index_name,
            model=model,
            min_df=1,
            tokenizer="word",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
            hyperparams={'b': 0.75, 'k1': 1.5}
        )
        sr.index(collection)
        return sr

    # def _merge_rankings(self, r1, r2):
    #     merged_ranking = {}
    #     for text_idx in set(r1).union(r2):
    #         if text_idx in r1 and text_idx in r2:
    #             merged_ranking[text_idx] = {x: r1[text_idx].get(x, 0) + r2[text_idx].get(x, 0) for x in
    #                                         set(r1[text_idx]).union(r2[text_idx])}
    #         elif text_idx in r1:
    #             merged_ranking[text_idx] = r1[text_idx]
    #
    #         elif text_idx in r2:
    #             merged_ranking[text_idx] = r2[text_idx]
    #         else:
    #             raise Exception("text_idx must be in r1 or r2.")
    #
    #     return merged_ranking

    def _get_ranking(self, bsearch_results, cls):
        ranking = {}
        for query_idx, docs in bsearch_results.items():
            test_sample_idx = int(query_idx.split("_")[-1])
            test_text_idx = self.samples[test_sample_idx]["text_idx"]
            labels_ids = {}
            for doc_idx, score in docs.items():
                idx = int(doc_idx.split("_")[-1])
                for label_idx in self.samples[idx]["labels_ids"]:
                    if cls in self.label_cls[label_idx]:
                        if self.params.retriever.sparse.aggregation == "sum":
                            labels_ids[f"label_{label_idx}"] = score + labels_ids.get(f"label_{label_idx}", 0)
                        elif self.params.retriever.sparse.aggregation == "max":
                            if score > labels_ids.get(f"label_{label_idx}", 0):
                                labels_ids[f"label_{label_idx}"] = score
                        else:
                            raise Exception("Aggregation must be set to be equals sum or max.")

            if cls in self.text_cls[test_text_idx]:
                if len(labels_ids) > 0:
                    ranking[f"text_{test_text_idx}"] = {k: v for k, v in
                                                        heapq.nlargest(self.params.retriever.sparse.num_labels,
                                                                       labels_ids.items(),
                                                                       key=lambda item: item[
                                                                           1])}  # dict(sorted(labels_ids.items(), key=lambda item: item[1], reverse=True)[:num_labels])
                else:
                    ranking[f"text_{test_text_idx}"] = {"label_-1": 0.0}

        return ranking

    def run(self):

        rankings = {}
        results = []
        for fold_idx in self.params.data.folds:
            rankings[fold_idx] = {}
            for split in ["test"]:
                rankings[fold_idx][split] = {}
                # get collection and queries by fold
                collection, queries = self._get_collection_and_queries(self.samples, fold_idx, split)

                rtvr = self._get_sparse_retriever(collection,
                                                  index_name=f"{self.params.data.name}_{fold_idx}_SR",
                                                  model="bm25")
                bsearch_results = rtvr.bsearch(
                    queries=queries,
                    cutoff=self.params.retriever.sparse.cutoff,
                    # batch_size bounds peak RAM during scoring (memory-only knob; does
                    # not change rankings). Needed for large corpora with long queries
                    # such as EURLEX-4K, where retriv's default of 1000 OOMs.
                    batch_size=self.params.retriever.sparse.get("batch_size", 1000),
                )

                # ranking by label cls
                for cls in self.params.eval.label_cls:
                    ranking = self._get_ranking(bsearch_results, cls)

                    result = evaluate(
                        Qrels({key: value for key, value in self.relevance_map.items() if key in ranking.keys()}),
                        Run(ranking),
                        metrics=self._get_metrics()
                    )
                    result = {k: round(v, 3) for k, v in result.items()}
                    result["fold_idx"] = fold_idx
                    result["split"] = split
                    result["cls"] = cls

                    rankings[fold_idx][split][cls] = ranking
                    results.append(result)
            self.checkpoint_ranking(rankings[fold_idx], fold_idx)
            self._checkpoint_results(results, fold_idx)

    def checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}BM25_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}BM25_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _checkpoint_results(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}BM25_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame(result).to_csv(
            f"{result_dir}BM25_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)
