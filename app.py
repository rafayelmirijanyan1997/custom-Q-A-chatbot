import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def main():
    st.set_page_config(page_title="Custom Q&A Chatbot")
    st.header("Chat with your PDFs")

    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Process"):
        raw_text = get_pdf_text(pdf_docs)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_texts([raw_text], embedding=embeddings)

        llm = OllamaLLM(model="llama3.2")

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        st.session_state.conversation = conversation

    if "conversation" in st.session_state:
        user_question = st.text_input("Ask a question:")
        if user_question:
            response = st.session_state.conversation({"question": user_question})
            st.write(response["answer"])


if __name__ == "__main__":
    load_dotenv()
    main()
