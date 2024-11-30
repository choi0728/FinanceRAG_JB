import csv
import json
import logging
import os
import asyncio
import nest_asyncio
import openai
import re
import tiktoken
import pandas as pd
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval
from tqdm import tqdm, trange

from financerag.common import Generator, HFDataLoader, Reranker, Retrieval
from financerag.tasks.TaskMetadata import TaskMetadata
from lancedb import connect
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import ColbertReranker

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

nest_asyncio.apply()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/AbsTask.py
class BaseTask:
    def __init__(self, metadata: TaskMetadata):
        self.metadata: TaskMetadata = metadata
        self.queries: Optional[Dict[str, str]] = None
        self.corpus: Optional[Dict[str, Dict[Literal["title", "text"], str]]] = None
        self.retrieve_results: Optional[Dict] = None
        self.rerank_results: Optional[Dict] = None
        self.generate_results: Optional[Dict] = None

        self.openai_embedder = get_registry().get("openai").create(name="text-embedding-ada-002")
        self.reranker = ColbertReranker()
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.hybrid_retriever = None
        self.table = None
        self.Schema = self._define_schema()

        self.client = openai

        self.punctuation_pattern = re.compile(r"[!\"#&*+,/:;<=>?@[\]^_`{|}~]")

        self.load_data()

    def _define_schema(self):
        class TextSchema(LanceModel):
            doc_id: str
            find_id: str
            title: str
            text: str = self.openai_embedder.SourceField()
            vector: Vector(self.openai_embedder.ndims()) = self.openai_embedder.VectorField()
        return TextSchema

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def load_data(self):
        if (self.corpus is None) or (self.queries is None):
            dataset_path = self.metadata_dict["dataset"]["path"]
            subset = self.metadata_dict["dataset"]["subset"]

            corpus, queries = HFDataLoader(
                hf_repo=dataset_path,
                subset=subset,
                keep_in_memory=False,
            ).load()

            self.queries = {query["id"]: query["text"] for query in queries}
            self.corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }

    def create_hybrid_retriever(self, mode: str = "overwrite", batch_size: int = 64):
        if self.hybrid_retriever is None:
            logger.info("Creating hybrid search table")
            db = connect("/tmp/.lancedb")
            self.hybrid_retriever = db.create_table("hybrid_search_table", schema=self.Schema, on_bad_vectors="drop", mode=mode)
            corpus_list = list(self.corpus.items())
            for i in trange(0, len(corpus_list), batch_size, desc="Adding corpus to hybrid search table"):
                batch = corpus_list[i:i+batch_size]
                texts = [self._clean_text("; ".join(doc_data["title"].split('_')) + "\n" + doc_data["text"]) for _, doc_data in batch]
                data = []
                embeddings = self.batch_get_embeddings(texts)
                for (doc_id, doc_data), embedding in zip(batch, embeddings):
                    data.append({
                        "doc_id": doc_id,
                        "find_id": doc_id[1:5],
                        "title": doc_data["title"],
                        "text": doc_data["text"],
                        "vector": embedding
                    })
                self.hybrid_retriever.add(data=data, on_bad_vectors="drop")
            try:
                self.hybrid_retriever.create_fts_index(['title', 'text'], replace=True)
            except Exception as e:
                logger.warning(f"Failed to create FTS index: {e}")

    def _clean_text(self, string: str, max_tokens: int = 8192) -> str:
        clean_string = string.encode('utf-8', 'replace').decode('utf-8', 'replace')
        clean_string = ' '.join(clean_string.split())
        toks = self.token_encoder.encode(clean_string)
        if len(toks) > max_tokens:
            clean_string = self.token_encoder.decode(toks[:max_tokens])
        return clean_string

    def _remove_punctuation(self, string: str) -> str:
        return self.punctuation_pattern.sub('', string)

    @lru_cache(maxsize=1000)
    def keyword_extraction_expansion(self, query: str, top_k: int = 100) -> List[str]:
        response = self.client.Completion.create(
            engine="davinci",
            prompt=f"Extract and expand search keywords from the following query, focusing on financial terms, companies, person names, and dates:\n\n{query}\n\nKeywords:",
            max_tokens=64,
            temperature=0.5,
            n=1,
            stop=None,
        )
        keywords = response.choices[0].text.strip()
        return [kw.strip() for kw in keywords.split(',')]

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.Embedding.create(input=batch, model="text-embedding-ada-002")
            batch_embeddings = [data['embedding'] for data in response['data']]
            embeddings.extend(batch_embeddings)
        return embeddings

    def _rename_score_column(self, df: pd.DataFrame, new_name: str) -> bool:
        if new_name in df.columns or len(df) == 0:
            return False

        if "_score" in df.columns:
            df.rename(columns={"_score": new_name}, inplace=True)
        elif "_relevance_score" in df.columns:
            df.rename(columns={"_relevance_score": new_name}, inplace=True)
        else:
            raise ValueError(f"No score column found in DataFrame columns: {df.columns}")
        return True

    async def async_hybrid_retrieve_rerank(self, top_k: int = 100, query_ids: Optional[List[str]] = None, **kwargs) -> Dict[str, Dict[str, float]]:
        logger.info("Starting asynchronous hybrid retrieval and reranking.")
        results = {}
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=10)

        async def process_query(q_id, query):
            return await loop.run_in_executor(executor, self.process_single_query, q_id, query, top_k)

        tasks = []
        for q_id, query in self.queries.items():
            if query_ids is not None and q_id not in query_ids:
                continue
            tasks.append(process_query(q_id, query))

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing queries"):
            q_id, result = await future
            results[q_id] = result

        return results

    def process_single_query(self, q_id, query, top_k):
        try:
            retrieved_docs = (self.hybrid_retriever
                        .search(query=query, query_type='hybrid')
                        .where(f"find_id = '{q_id[1:5]}' ", prefilter=True)
                        .rerank(reranker=self.reranker)
                        .limit(top_k)
                        .to_pandas()
                )
        except Exception as e:
            retrieved_docs = (self.hybrid_retriever
                            .search(query=query, query_type='vector')
                            .rerank(reranker=self.reranker)
                            .limit(top_k)
                            .to_pandas()
                )
        retrieved_docs.dropna(inplace=True)
        self._rename_score_column(retrieved_docs, "score")
        retrieved_docs = retrieved_docs.sort_values(by='score', ascending=False).drop_duplicates(subset=['doc_id'], keep='first').reset_index(drop=True)
        result = {doc_id: score for doc_id, score in zip(retrieved_docs['doc_id'], retrieved_docs['score'])}
        return q_id, result

    def hybrid_retrieve_rerank(self, top_k: int = 100, query_ids: Optional[List[str]] = None, **kwargs) -> Dict[str, Dict[str, float]]:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_hybrid_retrieve_rerank(top_k=top_k, query_ids=query_ids, **kwargs))
        return results

    def retrieve(
        self, retriever: Retrieval, top_k: Optional[int] = 100, **kwargs
    ) -> Dict[str, Dict[str, float]]:
        if not issubclass(type(retriever), Retrieval):
            raise TypeError(f"{type(retriever)} must be a subclass of the `Retrieval` class")

        if (self.corpus is None) or (self.queries is None):
            raise ValueError("Data has not been loaded.")

        self.retrieve_results = retriever.retrieve(
            queries=self.queries, corpus=self.corpus, top_k=top_k, **kwargs
        )

        return self.retrieve_results

    def rerank(
        self,
        reranker: Reranker,
        results: Optional[Dict[str, Dict[str, float]]] = None,
        top_k: int = 100,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if not issubclass(type(reranker), Reranker):
            raise TypeError(f"{type(reranker)} must be a subclass of the `Reranker` class")

        if (self.corpus is None) or (self.queries is None):
            raise ValueError("Data has not been loaded.")

        if results is None:
            if self.retrieve_results is not None:
                results = self.retrieve_results
            else:
                raise ValueError("Neither retrieve_results nor results can be None simultaneously.")

        self.rerank_results = reranker.rerank(
            queries=self.queries,
            corpus=self.corpus,
            results=results,
            top_k=top_k,
            batch_size=batch_size,
            **kwargs,
        )

        return self.rerank_results

    def generate(
        self,
        model: Generator,
        results: Optional[Dict] = None,
        prepare_messages: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, str]:
        if not issubclass(type(model), Generator):
            raise TypeError(f"{type(model)} must be a subclass of the `Generator` class")

        if prepare_messages is None:
            logger.info(
                "No prepare_messages function provided. "
                "Using default message preparation function, which selects the highest scored document for each query."
            )

            def default_messages(
                query: str, documents: List[Tuple[str, float]]
            ) -> List[Dict]:
                first_document = max(documents, key=lambda x: x[1])[0]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Document: {first_document}"
                                   f"\nGenerate an answer to the question from the document."
                                   f"\nQuestion: {query}",
                    },
                ]
                return messages

            prepare_messages = default_messages

        if results is None:
            results = (
                self.rerank_results
                if self.rerank_results is not None
                else self.retrieve_results
            )
            assert results is not None, (
                "Neither rerank_results nor retrieve_results are available. "
                "One of them must be provided."
            )

        messages_dict = self.prepare_generation_inputs(results, prepare_messages)
        self.generate_results = model.generation(messages_dict, **kwargs)

        return self.generate_results

    def prepare_generation_inputs(
        self, results, prepare_messages
    ) -> Dict[str, List[dict]]:
        if (self.corpus is None) or (self.queries is None):
            raise ValueError("Data has not been loaded.")

        messages_dict: Dict[str, List[Dict[str, str]]] = {}
        logger.info("Preparing generation inputs for %d queries.", len(results))
        for query_id, result in results.items():
            query = self.queries[query_id]
            documents = [
                (self.corpus[doc_id], score) for doc_id, score in result.items()
            ]
            messages = prepare_messages(query, documents)
            messages_dict[query_id] = messages

        logger.info("Successfully prepared generation inputs for all queries.")
        return messages_dict

    def save_results(self, top_k: int = 10, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            return
        output_dir = os.path.join(output_dir, self.metadata.name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Output directory set to: {output_dir}")

        csv_file_path = os.path.join(output_dir, "results.csv")
        jsonl_file_path = os.path.join(output_dir, "results_output.jsonl")
        logger.info(f"Saving top {top_k} results to CSV file: {csv_file_path}")

        final_result = (
            self.rerank_results
            if self.rerank_results is not None
            else self.retrieve_results
        )

        if final_result is not None:
            with open(jsonl_file_path, "w") as f:
                for q_id, doc_scores in final_result.items():
                    f.writelines(json.dumps({"query_id": q_id, "corpus_id": doc_id, "score": score}) + "\n"
                                 for doc_id, score in doc_scores.items())
            with open(csv_file_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["query_id", "corpus_id"])
                logger.info("Writing header ['query_id', 'corpus_id'] to CSV.")

                for q_id, doc_scores in final_result.items():
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda item: item[1], reverse=True
                    )[:top_k]

                    for doc_id, _ in sorted_docs:
                        writer.writerow([q_id, doc_id])

            logger.info(f"Top {top_k} results saved successfully to {csv_file_path}")

        if self.generate_results is not None:
            jsonl_file_path = os.path.join(output_dir, "output.jsonl")
            logger.info(f"Saving generate_results to JSONL file: {jsonl_file_path}")

            with open(jsonl_file_path, "w") as f:
                f.writelines(
                    json.dumps({"query_id": q_id, "answer": answer}) + "\n"
                    for q_id, answer in self.generate_results.items()
                )

            logger.info(f"generate_results saved successfully to {jsonl_file_path}")

    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                'For evaluation, we ignore identical query and document ids (default), '
                'please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            for qid, rels in results.items():
                results[qid] = {pid: score for pid, score in rels.items() if qid != pid}

        filtered_results = {qid: rels for qid, rels in results.items() if qid in qrels}

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(filtered_results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        num_queries = len(scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / num_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / num_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / num_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / num_queries, 5)

        logger.info("\nEvaluation Results:")
        for metric in [ndcg, _map, recall, precision]:
            for k in metric.keys():
                logger.info(f"{k}: {metric[k]:.4f}")
            logger.info("\n")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Optionally save detailed evaluation results
            # For example, saving per-query metrics

        return ndcg, _map, recall, precision
