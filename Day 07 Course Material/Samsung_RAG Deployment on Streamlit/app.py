import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------

st.set_page_config(
    page_title="Samsung Washing Machine Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Samsung Washing Machine Assistant")
st.markdown(
    "Ask any question about the Samsung Washing Machine Manual."
)

# --------------------------------------------------------
# OpenAI API Key
# --------------------------------------------------------

st.sidebar.title("🔑 OpenAI API Key")

api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-..."
)

if not api_key:
    st.info("👈 Please enter your OpenAI API Key in the sidebar to begin.")
    st.stop()

# --------------------------------------------------------
# Cache Resources
# --------------------------------------------------------

@st.cache_resource
def load_rag():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant for Samsung Washing Machine documentation.

Answer ONLY using the provided context.

If the answer cannot be found, reply:

"I don't know based on the provided documentation."

Keep answers concise.

Question:
{question}

Context:
{context}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


rag_chain = load_rag()

# --------------------------------------------------------
# Chat History
# --------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------
# Chat Input
# --------------------------------------------------------

question = st.chat_input("Ask your question here...")

if question:

    # User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant
    with st.chat_message("assistant"):

        with st.spinner("Searching documentation..."):

            response = rag_chain.invoke(question)

            answer = response.content

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )