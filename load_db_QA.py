import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables (OPENAI_API_KEY from .env)
load_dotenv()

# Set Streamlit page settings
st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")
st.title("🤖 A Glass of Milk Chapter")

# Load the FAISS retriever from local vectorstore
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_retriever()

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user prompt
prompt = st.chat_input("Ask a question about your PDF...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        try:
            response = qa.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# 💾 Download full conversation as .txt
if len(st.session_state.messages) > 0:
    full_chat = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Bot"
        full_chat += f"{role}: {msg['content']}\n\n"

    st.download_button(
        label="💾 Download Full Conversation",
        data=full_chat,
        file_name="chat_history.txt",
        mime="text/plain"
    )
