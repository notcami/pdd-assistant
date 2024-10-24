import os
import shutil
import pdfplumber
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
import nltk

from query_data import PROMPT_TEMPLATE

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

#CHROMA_PATH = "chroma"
DATA_PATH = "data"

GITHUB_API_URL = "https://api.github.com/repos/notcami/pdd-assistant/contents/data"
GITHUB_REPO_URL = "https://raw.githubusercontent.com/notcami/pdd-assistant/main/data/"

def get_pdf_files_from_github():
    response = requests.get(GITHUB_API_URL)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['name'] for file in files if file['name'].endswith('.pdf')]
        return pdf_files
    else:
        st.error(f"Не удалось получить список файлов. Код ошибки: {response.status_code}")
        return []

def download_file_from_github(filename, save_path):
    url = f"{GITHUB_REPO_URL}{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        st.write(f"Downloaded {filename} successfully.")
    else:
        st.error(f"Failed to download {filename}. Status code: {response.status_code}")

def download_all_pdfs():
    os.makedirs(DATA_PATH, exist_ok=True)
    pdf_files = get_pdf_files_from_github()
    if pdf_files:
        for pdf in pdf_files:
            download_file_from_github(pdf, os.path.join(DATA_PATH, pdf))
    else:
        st.error("No PDF files found to download.")

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(DATA_PATH, filename)
            st.write(f"Processing file: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def extract_text_from_pdf(pdf_document):
    with pdfplumber.open(pdf_document) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']), persist_directory=CHROMA_PATH
    )
    st.write(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_database(query_text):
    embedding_function = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    return results

def generate_response(context_text, question):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text

st.title("PDF Query Assistant")

if not os.listdir(DATA_PATH):
    st.write("Downloading PDF files from GitHub...")
    download_all_pdfs()
# else:
#     st.write("PDF files are already downloaded.")

if not os.path.exists(CHROMA_PATH):
    st.write("Generating database...")
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    st.success("Database generated successfully!")

query_text = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if query_text:
        results = query_database(query_text)
        if len(results) == 0 or results[0][1] < 0.7:
            st.warning("Unable to find matching results.")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            response = generate_response(context_text, query_text)
            sources = [doc.metadata.get("source", None) for doc, _ in results]

            response_paragraphs = response.split('\n\n')
            st.markdown("### Response:")
            for paragraph in response_paragraphs:
                st.write(paragraph.strip())

            st.markdown("### Sources:")
            unique_sources = set(sources)
            for source in unique_sources:
                st.write(f"- {source}")
