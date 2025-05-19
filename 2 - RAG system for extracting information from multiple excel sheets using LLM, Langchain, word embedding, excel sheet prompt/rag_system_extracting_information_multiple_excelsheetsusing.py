 
!pip install langchain chromadb pandas -q

!pip install langchain_huggingface -q

!pip install langchain_community -q

import os
import pandas as pd

#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from langchain_community.vectorstores import Chroma

# 1. Load and preprocess Excel sheets

excel_files = ["/content/sample_data/insurance-dataset.xlsx", "/content/sample_data/Sports Data.xlsx"]  # Add your Excel file paths here

all_docs = []



for file in excel_files:

    xls = pd.ExcelFile(file)

    for sheet_name in xls.sheet_names:

        df = xls.parse(sheet_name)

        for idx, row in df.iterrows():

            doc_text = f"File: {file}, Sheet: {sheet_name}, Row: {row.to_dict()}"

            all_docs.append(doc_text)



# 2. Split documents if needed (optional for large rows)

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

split_docs = []

for doc in all_docs:

    split_docs.extend(text_splitter.create_documents([doc]))



# 3. Create embeddings and vector store using HuggingFace

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vectorstore = FAISS.from_documents(split_docs, embeddings)
CHROMA_PATH = "Chroma"
# embed the chunks as vectors and load them into the database.
db_chroma = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_PATH)


# 4. Set up the HuggingFace LLM (e.g., Falcon, Llama, or any supported model)

llm_pipeline = pipeline(

    "text-generation",

    model="google/flan-t5-base",  # You can use any supported HuggingFace model here

    max_new_tokens=256,

    do_sample=True,

    temperature=0.2,

)

llm = HuggingFacePipeline(pipeline=llm_pipeline)



qa_chain = RetrievalQA.from_chain_type(

    llm=llm,

    retriever=db_chroma.as_retriever()

)



# 5. Ask a question

question = "list of matches that happened on night"

answer = qa_chain.run(question)

print("Q:", question)

print("A:", answer)