import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

from rag import (DEFAULT_TOP_K, LLM_MODEL, build_index, chunk_documents,
                 get_embeddings, load_pdf_pages)


def main():
    st.set_page_config(page_title="Custom Q&A Chatbot")
    st.header("Chat with your PDFs")

    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Process") and pdf_docs:
        with st.spinner("Chunking and indexing..."):
            pages = load_pdf_pages(pdf_docs)
            chunks = chunk_documents(pages)
            index = build_index(chunks, get_embeddings("ollama"))
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=OllamaLLM(model=LLM_MODEL),
            retriever=index.as_retriever(search_kwargs={"k": DEFAULT_TOP_K}),
            memory=memory,
            return_source_documents=True,
        )
        st.success(f"Indexed {len(pages)} pages as {len(chunks)} chunks.")

    if "conversation" in st.session_state:
        user_question = st.text_input("Ask a question:")
        if user_question:
            response = st.session_state.conversation({"question": user_question})
            st.write(response["answer"])
            with st.expander("Sources"):
                for doc in response["source_documents"]:
                    st.markdown(f"**{doc.metadata['source']}, page {doc.metadata['page']}**")
                    st.caption(doc.page_content[:400])


if __name__ == "__main__":
    main()
