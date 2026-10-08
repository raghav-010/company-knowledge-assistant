"""
eval_ragas.py — RAGAS evaluation for Company Knowledge Assistant
Raghav Balaji V | github.com/raghav-010

Measures 4 RAG quality metrics:
  faithfulness       — is the answer grounded in the retrieved context?
  answer_relevancy   — does the answer address the question?
  context_precision  — are the retrieved chunks relevant to the question?
  context_recall     — does the retrieved context cover the reference answer?

Usage (run from project root, Docker must be up):
  python -m app.eval_ragas

NOTE: Uses Gemini 2.5 Flash for evaluation (same as the RAG pipeline).
Run this AFTER the RAG pipeline is working and docs are ingested.
"""
import asyncio
import json
import requests

from ragas import evaluate
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate_rag_system(test_path: str = "seed/qna_test.json"):
    test_data = load_jsonl(test_path)
    results = []

    for item in test_data:
        question = item["question"]
        reference_answer = item["answer"]

        res = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
        ).json()

        answer = res["answer"]
        contexts = res["contexts"]

        results.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference_answer,
            )
        )

    ds = EvaluationDataset(results)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Using Gemini for eval (no OpenAI credits needed)
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    run_config = RunConfig(max_workers=8, timeout=60)
    eval_result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=eval_llm,
        run_config=run_config,
    )

    print("\n=== RAGAS Evaluation Results ===")
    df = eval_result.to_pandas()
    print(df.mean(numeric_only=True))
    return df


if __name__ == "__main__":
    asyncio.run(evaluate_rag_system())
