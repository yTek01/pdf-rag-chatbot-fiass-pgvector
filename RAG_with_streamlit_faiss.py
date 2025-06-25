
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda

from langchain_postgres.vectorstores import PGVector
from database import COLLECTION_NAME, CONNECTION_STRING
from langchain_community.storage import RedisStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from IPython.display import display, HTML
from base64 import b64decode
import os, hashlib, shutil, uuid, json, time
import torch, redis, streamlit as st
import logging


from dotenv import load_dotenv
load_dotenv()

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
logging.basicConfig(level=logging.INFO)
client = redis.Redis(host="localhost", port=6379, db=0)


def load_pdf_data(file_path):
    logging.info(f"Data ready to be partitioned and loaded ")
    raw_pdf_elements = partition_pdf(
        filename=file_path,
      
        infer_table_structure=True,
        strategy = "hi_res",
        
        extract_image_block_types = ["Image"],
        extract_image_block_to_payload  = True,

        chunking_strategy="by_title",     
        mode='elements',
        max_characters=10000,
        new_after_n_chars=5000,
        combine_text_under_n_chars=2000,
        image_output_dir_path="data/",
    )
    logging.info(f"Pdf data finish loading, chunks now available!")
    return raw_pdf_elements

def get_pdf_hash(pdf_path):
    """Generate a SHA-256 hash of the PDF file content."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.sha256(pdf_bytes).hexdigest()

def summarize_text_and_tables(text, tables):
    logging.info("Ready to summarize data with LLM")
    prompt_text = """You are an assistant tasked with summarizing text and tables. \
    
                    You are to give a concise summary of the table or text and do nothing else. 
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
    summarize_chain = {"element": RunnablePassthrough()}| prompt | model | StrOutputParser()
    logging.info(f"{model} done with summarization")
    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }
  
def initialize_retriever():
    store = RedisStore(client=client)
    id_key = "doc_id"
    embeddings = OpenAIEmbeddings()
    faiss_index_path = "faiss_index_dir"

    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logging.info("Loaded existing FAISS index from disk.")
        logging.info(f"FAISS contains {len(vectorstore.index_to_docstore_id)} indexed vectors")
    else:
        logging.warning("No FAISS index found yet. Will be created after document ingestion.")
        vectorstore = None 

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    return retriever


def try_load_faiss():
    embeddings = OpenAIEmbeddings()
    faiss_index_path = "faiss_index_dir"

    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logging.info("Loaded existing FAISS index from disk.")
        logging.info(f"FAISS contains {len(vectorstore.index_to_docstore_id)} indexed vectors")
        return vectorstore
    else:
        logging.warning("No FAISS index found yet. Will be created after document ingestion.")
        return None


def store_docs_in_retriever(text_chunks, text_summaries, table_chunks, table_summaries, retriever):
    all_summary_docs = []
    all_source_docs = []
    all_doc_ids = []
    all_chunks = [(text_chunks, text_summaries), (table_chunks, table_summaries)]
    embeddings = OpenAIEmbeddings()

    for chunks, summaries in all_chunks:
        if not summaries:
            continue

        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        all_summary_docs.extend(summary_docs)
        all_doc_ids.extend(doc_ids)
        all_source_docs.extend(chunks)

    if retriever.vectorstore is None:
        retriever.vectorstore = FAISS.from_documents(
            documents=all_summary_docs,
            embedding=embeddings,
            ids=all_doc_ids
        )
        logging.info("Created new FAISS index from documents.")
    else:
        retriever.vectorstore.add_documents(all_summary_docs, ids=all_doc_ids)

    retriever.docstore.mset(list(zip(all_doc_ids, all_source_docs)))

    logging.info(f"Stored {len(all_summary_docs)} vectors in FAISS")
    logging.info(f"Stored {len(all_source_docs)} original docs in Redis")

    retriever.vectorstore.save_local("faiss_index_dir")
    logging.info("FAISS index saved to disk.")
    
    return retriever

def parse_retriver_output(data):
    parsed_elements = []
    for element in data:
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        
        parsed_elements.append(element)
    
    return parsed_elements


def chat_with_llm(retriever):

    logging.info(f"Context ready to send to LLM ")
    prompt_text = """
                You are an AI Assistant tasked with understanding detailed
                information from text and tables. You are to answer the question based on the 
                context provided to you. You must not go beyond the context given to you.
                
                Context:
                {context}

                Question:
                {question}
                """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
 
    rag_chain = ({
       "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
        } 
        | prompt 
        | model 
        | StrOutputParser()
        )
        
    logging.info(f"Completed! ")

    return rag_chain

def _get_file_path(file_upload):
    """
    Accepts either a path (str) or a Streamlit file uploader object.
    Returns a string path to the file.
    """
    if isinstance(file_upload, str) and os.path.exists(file_upload):
        return file_upload

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    if hasattr(file_upload, "name"):
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path

    raise ValueError("Invalid file_upload input. Must be a valid path or Streamlit upload object.")


def process_pdf(file_upload):
    print('Processing PDF hash info...')
    
    file_path = _get_file_path(file_upload)
    pdf_hash = get_pdf_hash(file_path)

    vectorstore = try_load_faiss()
    existing = client.exists(f"pdf:{pdf_hash}")
    print(f"Checking Redis for hash {pdf_hash}: {'Exists' if existing else 'Not found'}")

    if existing and vectorstore:
        print(f"PDF already exists with hash {pdf_hash}. Skipping upload.")
        return MultiVectorRetriever(vectorstore=vectorstore, docstore=RedisStore(client=client), id_key="doc_id")

    print(f"New PDF detected. Processing... {pdf_hash}")
    pdf_elements = load_pdf_data(file_path)

    tables = [element.metadata.text_as_html for element in pdf_elements if 'Table' in str(type(element))]
    text = [element.text for element in pdf_elements if 'CompositeElement' in str(type(element))]

    summaries = summarize_text_and_tables(text, tables)

    embeddings = OpenAIEmbeddings()
    doc_ids = [str(uuid.uuid4()) for _ in summaries['text']]
    summary_docs = [
        Document(page_content=summary, metadata={"doc_id": doc_ids[i]})
        for i, summary in enumerate(summaries['text'])
    ]

    vectorstore = FAISS.from_documents(summary_docs, embedding=embeddings, ids=doc_ids)
    RedisStore(client=client).mset(list(zip(doc_ids, text)))
    vectorstore.save_local("faiss_index_dir")
    logging.info("FAISS index created and saved to disk.")

    client.set(f"pdf:{pdf_hash}", json.dumps({"text": "PDF processed"}))
    
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=RedisStore(client=client), id_key="doc_id")



def invoke_chat(file_upload, message):

    retriever = process_pdf(file_upload)
    rag_chain = chat_with_llm(retriever)
    response = rag_chain.invoke(message)
    response_placeholder = st.empty()
    response_placeholder.write(response)
    return response


def main():
  
    st.title("PDF Chat Assistant ")
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

 
    file_upload = st.sidebar.file_uploader(
    label="Upload", type=["pdf"], 
    accept_multiple_files=False,
    key="pdf_uploader"
    )

    if file_upload:     
        st.success("File uploaded successfully! You can now ask your question.")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response...")
            with st.spinner("Processing..."):
                user_message = " ".join([msg["content"] for msg in st.session_state.messages if msg])           
                response_message = invoke_chat(file_upload, user_message)
        
                duration = time.time() - start_time
                response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"

                st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
                st.write(f"Duration: {duration:.2f} seconds")
                logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")


    



if __name__ == "__main__":
    main()
	
	
	