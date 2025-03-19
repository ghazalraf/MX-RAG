import streamlit as st
import openai
import os
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter 
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "MyCustomAgent/1.0"

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define model
# model_local = OllamaLLM(model="mistral")
model_local = ChatOpenAI(model="gpt-3.5-turbo")

# Input URLs
urls = [
    'https://documentation.meraki.com/General_Administration/Cross-Platform_Content/Alerts_and_Notifications/Dashboard_Alerts_-_Connectivity_Issues',
    'https://documentation.meraki.com/MX/Site-to-site_VPN/Site-to-Site_VPN_Settings',
    'https://documentation.meraki.com/MX/Firewall_and_Traffic_Shaping/MX_Firewall_Settings',
    'https://documentation.meraki.com/MX/Client_VPN/AnyConnect_on_the_MX_Appliance',
    'https://documentation.meraki.com/MX/Client_VPN/AnyConnect_on_the_MX_Appliance/AnyConnect_Licensing_on_the_MX',
    'https://documentation.meraki.com/MX/Client_VPN/AnyConnect_on_the_MX_Appliance/AnyConnect_on_ASA_vs._MX',
]

# Function to process input with RAG
def process_input(question, retriever):
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Please provide the relevant sources of the information along with the answer.
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

    # Remove unwanted phrases like "Dear Valued Customer" and "Best Regards"
    response = response.replace("Dear Valued Customer", "").replace("Best Regards", "").strip()
    return response

# Streamlit UI
st.title("Enter Case Information:")

subject = st.text_input("Subject")
product = st.text_input("Product")
case_description = st.text_area("Case Description")

# Create a button to trigger processing
if st.button("Submit"):
    with st.spinner("Processing..."):
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split the text into chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)

        # Convert text chunks into embeddings and store in vector database
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Combine inputs into a single query
        user_query = f"Subject: {subject}\nProduct: {product}\nCase Description: {case_description}"
        response = process_input(user_query, retriever)

        # Display response
        st.write(response)
