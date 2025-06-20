import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Cache model loading
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        device_map="auto",
        torch_dtype=torch.float32
    )
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.5
    )
    return HuggingFacePipeline(pipeline=pipe)

# Custom prompt
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

custom_prompt_template = """
Use the pieces of information provided in the question to answer the user's question.
If you don't know the answer, just say that you don't know and don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

# Initialize Retrieval QA Chain
@st.cache_resource
def get_qa_chain():
    vectorstore = get_vectorstore()
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
    )
    return qa_chain

# Streamlit UI
def main():
    st.title("🩺 Ask Medibot - Your Medical QA Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask Medibot your medical question here...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            qa_chain = get_qa_chain()
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            sources_text = "\n\n".join(
                [f"📄 **Source {i+1}:**\n{doc.page_content}" for i, doc in enumerate(source_documents)]
            )

            full_response = f"{result}\n\n---\n**References:**\n{sources_text}"
            st.chat_message("assistant").markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
