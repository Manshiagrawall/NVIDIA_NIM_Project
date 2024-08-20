# import streamlit as st
# import os
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=NVIDIAEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
#         print("hEllo")
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


# st.title("Nvidia NIM Demo")
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")


# import streamlit as st
# import os
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Load the Groq API key
# os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# # Function to create vector embeddings and FAISS index
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = NVIDIAEmbeddings()

#         # Load documents
#         st.session_state.loader = PyPDFDirectoryLoader("./us_census")
#         st.session_state.docs = st.session_state.loader.load()
        
#         # Ensure documents were loaded
#         if len(st.session_state.docs) == 0:
#             st.error("No documents were loaded. Please check the document directory.")
#             return

#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

#         # Check embedding generation
#         try:
#             example_embedding = st.session_state.embeddings.embed_documents([st.session_state.final_documents[0].page_content])
#             print("Example Embedding Shape:", len(example_embedding[0]))
#         except Exception as e:
#             st.error(f"Error generating embeddings: {str(e)}")
#             return
        
#         # Try to create the FAISS index
#         try:
#             st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         except Exception as e:
#             st.error(f"An error occurred while creating the FAISS index: {str(e)}")

# # Streamlit app title
# st.title("Nvidia NIM Demo")

# # Load the LLM model
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# # Create the prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# # Input for the user question
# prompt1 = st.text_input("Enter Your Question From Documents")

# # Button to create document embeddings
# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# # Process the user query if provided
# if prompt1:
#     # Create the document chain and retriever
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     # Measure response time
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': prompt1})
#     print("Response time:", time.process_time() - start)
    
#     # Display the response
#     st.write(response['answer'])

#     # With a Streamlit expander, show document similarity search results
#     with st.expander("Document Similarity Search"):
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")

import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Function to create vector embeddings and FAISS index
def vector_embedding():
    try:
        if "vectors" not in st.session_state:
            st.session_state.embeddings = NVIDIAEmbeddings()

            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()
            
            # Ensure documents were loaded
            if len(st.session_state.docs) == 0:
                st.error("No documents were loaded. Please check the document directory.")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

            # Generate embeddings and create FAISS index
            try:
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            except Exception as e:
                st.error(f"An error occurred while creating the FAISS index: {str(e)}")
                return

            st.success("Vector Store DB is ready!")
    except Exception as e:
        st.error(f"An unexpected error occurred during vector embedding: {str(e)}")

# Streamlit app title
st.title("Nvidia NIM Demo")

# Load the LLM model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Input for the user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to create document embeddings
if st.button("Documents Embedding"):
    vector_embedding()

# Ensure that vectors have been successfully created before proceeding
if prompt1:
    if "vectors" in st.session_state:
        try:
            # Create the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Measure response time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            print("Response time:", time.process_time() - start)
            
            # Display the response
            st.write(response['answer'])

            # With a Streamlit expander, show document similarity search results
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"An error occurred during retrieval: {str(e)}")
    else:
        st.error("Vector embeddings are not set. Please run the 'Documents Embedding' step first.")
