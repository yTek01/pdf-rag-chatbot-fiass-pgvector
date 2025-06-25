
from evaluation import evaluate_pipeline, evaluate_pipeline_faiss
from evaluation_data import evaluation_dataset_hbs, evaluation_dataset_layout, evaluation_dataset_edu
from aggregate_results import summarize_results

pdf_path = "data/hbspapers_48__1.pdf"
pdf_path_2 = "data/layout-parser-paper.pdf"
pdf_path_3 = "data/Retrieval-Augmented_Generation_RAG_Chatbots_for_Ed.pdf"

faiss_results = evaluate_pipeline_faiss(pdf_path, evaluation_dataset_hbs, pipeline_name="FAISS")
print("Process with FAISS: hbspapers_48__1.pdf")
summarize_results(faiss_results)

pgvector_results = evaluate_pipeline(pdf_path, evaluation_dataset_hbs, pipeline_name="PGVector")
print("Process with PGVector: hbspapers_48__1.pdf")
summarize_results(pgvector_results)


faiss_results = evaluate_pipeline_faiss(pdf_path_2, evaluation_dataset_layout, pipeline_name="FAISS")
print("Process with FAISS layout-parser-paper.pdf")
summarize_results(faiss_results)

pgvector_results = evaluate_pipeline(pdf_path_2, evaluation_dataset_layout, pipeline_name="PGVector")
print("Process with PGVector layout-parser-paper.pdf")
summarize_results(pgvector_results)


faiss_results = evaluate_pipeline_faiss(pdf_path_3, evaluation_dataset_edu, pipeline_name="FAISS")
print("Process with FAISS Retrieval-Augmented_Generation_RAG_Chatbots_for_Ed.pdf")
summarize_results(faiss_results)

pgvector_results = evaluate_pipeline(pdf_path_3, evaluation_dataset_edu, pipeline_name="PGVector")
print("Process with PGVector Retrieval-Augmented_Generation_RAG_Chatbots_for_Ed.pdf")
summarize_results(pgvector_results)
