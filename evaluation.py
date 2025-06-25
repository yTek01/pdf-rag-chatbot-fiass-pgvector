
from RAG_with_streamlit_pg_vec import process_pdf, chat_with_llm
from RAG_with_streamlit_faiss import process_pdf as pf, chat_with_llm as cl
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import numpy as np


def compute_hit_mrr(retrieved_chunks, expected_answer):
    """
    Compute Hit@1 and MRR for a list of retrieved chunks.

    Parameters:
    - retrieved_chunks: list of LangChain Document objects or raw byte content
    - expected_answer: string (ground-truth answer)

    Returns:
    - hit (int): 1 if answer was found in any chunk, else 0
    - rr (float): Reciprocal rank score
    """
    hit = 0
    rr = 0.0

    for i, chunk in enumerate(retrieved_chunks):
        content = (
            chunk.page_content if hasattr(chunk, "page_content") else
            chunk.decode("utf-8") if isinstance(chunk, bytes) else
            str(chunk)
        )

        if expected_answer.lower() in content.lower():
            hit = 1
            rr = 1 / (i + 1)
            break

    return hit, rr

def compute_semantic_similarity(answer1, answer2):
    """
    Compute cosine similarity between two answers using OpenAI embeddings.
    Returns a float between 0 and 1.
    """
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([answer1, answer2])
    v1, v2 = np.array(vectors[0]), np.array(vectors[1])
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(cosine_sim)


def score_response_with_llm(question, gold_answer, model_answer):
    prompt = f"""
        You are an expert judge evaluating AI-generated answers.

        Question: {question}

        Reference Answer: {gold_answer}

        AI Answer: {model_answer}

        Rate on the following:

        Correctness (0–5): Does the AI answer factually align with the reference?
        Relevance (0–10): Is the AI answer appropriate for the question?
        Faithfulness (0–10): Does the AI answer stay true to the retrieved context?

        Return only JSON in this format:
        {{
        "correctness": <score>,
        "relevance": <score>,
        "faithfulness": <score>
        }}
    """
    model = ChatOpenAI(model="gpt-4", temperature=0)
    output = model.invoke(prompt)
    try:
        return eval(output.content) 
    except Exception:
        return {"correctness": 0, "relevance": 0, "faithfulness": 0}


def evaluate_pipeline(file_upload, evaluation_dataset, pipeline_name="PGVector"):
    print(f"Evaluating pipeline: {pipeline_name}")
    retriever = process_pdf(file_upload)
    rag_chain = chat_with_llm(retriever)

    results = []
    for pair in evaluation_dataset:
        question = pair["question"]
        gold_answer = pair["answer"]

        retrieved_chunks = retriever.get_relevant_documents(question)
        response = rag_chain.invoke(question)
        hit, rr = compute_hit_mrr(retrieved_chunks, gold_answer)
        judgment = score_response_with_llm(question, gold_answer, response)
        similarity_score = compute_semantic_similarity(gold_answer, response)
        results.append({
            "question": question,
            "response": response,
            "hit": hit,
            "mrr": rr,
            "semantic_similarity": similarity_score,
            **judgment
        })

    return results


def evaluate_pipeline_faiss(file_upload, evaluation_dataset, pipeline_name="PGVector"):
    print(f"Evaluating pipeline: {pipeline_name}")
    retriever = pf(file_upload)
    rag_chain = cl(retriever)

    results = []
    for pair in evaluation_dataset:
        question = pair["question"]
        gold_answer = pair["answer"]

        retrieved_chunks = retriever.get_relevant_documents(question)
        response = rag_chain.invoke(question)
        hit, rr = compute_hit_mrr(retrieved_chunks, gold_answer)
        judgment = score_response_with_llm(question, gold_answer, response)
        similarity_score = compute_semantic_similarity(gold_answer, response)
        results.append({
            "question": question,
            "response": response,
            "hit": hit,
            "mrr": rr,
            "semantic_similarity": similarity_score,
            **judgment
        })

    return results

