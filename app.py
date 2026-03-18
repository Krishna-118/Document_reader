import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils import process_pdfs
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Academic Textbook RAG", layout="wide")
st.title("📚 Academic Textbook Assistant")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload Textbook PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = process_pdfs(uploaded_files)
        st.success("Documents processed successfully!")
    else:
        st.warning("Please upload at least one PDF.")

# Chat input
query = st.text_input("Ask a textbook-based question")

if query and st.session_state.vectorstore:

    # ✅ Correct LLM setup
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],  # IMPORTANT
        model_name="llama3-70b-8192",
        temperature=0
    )

    # -----------------------------
    # Contextualize question prompt
    # -----------------------------
    contextualize_q_system_prompt = """
Given a chat history and the latest user question,
formulate a standalone question that can be understood
without chat history. Do NOT answer the question.
Return the reformulated question only.
"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("human", "{input}"),
        ]
    )

    # Retriever
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 1}
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # -----------------------------
    # QA Prompt (FIXED)
    # -----------------------------
    qa_system_prompt = """
You are a strict academic assistant.

You must answer ONLY using the provided textbook context.

STRICT RULES:
1. Use only the given context.
2. Do NOT add outside knowledge.
3. Preserve textbook terminology exactly.
4. Maintain formal academic tone.
5. Structure the answer like a 15-mark university exam answer.
6. If the answer is not found in the context, say:
   "The answer is not available in the provided textbook content."
7. Every statement must be traceable to the context.
8. Do not simplify or paraphrase unnecessarily.
"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", """
Context:
{context}

Question:
{input}

Instructions:
- Extract and reproduce the answer strictly from the context.
- Use sentences as they appear in the context wherever possible.
- Do not generate new explanations.
- Format the answer in clear academic paragraphs suitable for a 15-mark university answer.
"""),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    # ✅ FIXED chat history handling
    chat_history_messages = []

    for q, a in st.session_state.chat_history:
        if q and a:
            chat_history_messages.append(HumanMessage(content=str(q)))
            chat_history_messages.append(AIMessage(content=str(a)))

    # limit size
    chat_history_messages = chat_history_messages[-6:]

    # Invoke
    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history_messages
    })

    # Save history
    st.session_state.chat_history.append((query, result["answer"]))

    # Display answer
    st.markdown("### 📘 Answer")
    st.write(result["answer"])

    # Show sources
    with st.expander("📚 Source Documents"):
        for doc in result["context"]:
            st.write(doc.metadata)

else:
    st.info("Upload and process documents to start.")
