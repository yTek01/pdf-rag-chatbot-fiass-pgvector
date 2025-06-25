import numpy as np

def summarize_results(results):
    num = len(results)
    hr = np.mean([r["hit"] for r in results])
    mrr = np.mean([r["mrr"] for r in results])
    correctness = np.mean([r["correctness"] for r in results])
    relevance = np.mean([r["relevance"] for r in results])
    faithfulness = np.mean([r["faithfulness"] for r in results])
    semantic_similarity = np.mean([r["semantic_similarity"] for r in results])
    print("\n=== Evaluation Summary ===")
    print(f"Hit Rate (HR): {hr:.3f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print(f"Mean Correctness Score: {correctness:.2f} / 5")
    print(f"Mean Relevance Score: {relevance:.2f} / 10")
    print(f"Mean Faithfulness Score: {faithfulness:.2f} / 10")
    print(f"Mean Semantic Similarity: {semantic_similarity:.3f} (Cosine)")
