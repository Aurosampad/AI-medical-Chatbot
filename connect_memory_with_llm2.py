from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import torch  # Required for torch_dtype and device_map

# Load FLAN-T5 LLM Locally
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        device_map="auto",  # Automatically uses GPU if available
        torch_dtype=torch.float32  # Use float32 to avoid issues if no GPU
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Custom prompt template
custom_prompt_template = """
Use the pieces of information provided in the question to answer the user's question.
If you don't know the answer, just say that you don't know and don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
)

# Ask user query
user_query = input("Write Query here: ")
response = qa_chain.invoke({'query': user_query})

# Output result
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])

