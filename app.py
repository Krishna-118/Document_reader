import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import process_pdfs
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Document reader Chatbot", layout="wide")
st.title("📄 Document reader  Chatbot")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload document PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process document"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = process_pdfs(uploaded_files)
        st.success("documents processed successfully!")
    else:
        st.warning("Please upload at least one PDF.")

# Chat section
query = st.text_input("Ask questions about documents")

if query and st.session_state.vectorstore:
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # History aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.vectorstore.as_retriever(), contextualize_q_prompt
    )

    # QA prompt
    qa_system_prompt = (You are a strict academic assistant.

"Your task is to answer ONLY using the provided context from the textbook.

Rules:
1. Use ONLY the information given in the context.
2. Do NOT add external knowledge.
3. Do NOT simplify the language.
4. Maintain textbook terminology and formal academic tone.
5. Structure the answer clearly with headings and subheadings if present in the context.
6. If the answer is not found in the context, reply:
   "The answer is not available in the provided textbook content."
7. Do not include explanations outside the textbook material.
8. Do not summarize unless the context itself summarizes."

"Your answer must look like a university 15-mark exam answer."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Question answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Convert chat history to messages
    chat_history_messages = []
    for q, a in st.session_state.chat_history:
        chat_history_messages.append(HumanMessage(content=q))
        chat_history_messages.append(AIMessage(content=a))

    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history_messages
    })

    st.session_state.chat_history.append((query, result["answer"]))

    st.markdown("### 💡 Answer")
    st.write(result["answer"])

    with st.expander("📚 Source Documents"):
        for doc in result["context"]:
            st.write(doc.metadata)

else:
    st.info("Upload and process document to start chatting.")



