import streamlit as st

from streamlit_extras.let_it_rain import rain

from streamlit_extras.colored_header import colored_header

from streamlit_card import card

from annotated_text import annotated_text, annotation

from PIL import Image

from langchain_together import ChatTogether
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain_openai import OpenAIEmbeddings

import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]


embeddings = OpenAIEmbeddings()

EMBEDDING_DIM = 768

text_file = open("few_contexts.txt", "r")
text_content = text_file.read()
text_file.close()

# docs = text_content.split("\n ")
docs = text_content.split("***")


llm = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

qdrant = QdrantVectorStore.from_texts(
    docs,
    embedding=embeddings,
    location=":memory:",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.DENSE,
)


def search(question):

    return qdrant.similarity_search(question)


st.set_page_config(
    page_title="Anaaya",
)


image = Image.open("healthcare128.png")

left_co, cent_co, last_co = st.columns(3)

with cent_co:

    st.image(image)


colored_header(
    label="Anaaya",
    description="Unlocking Safer Healthcare Decisions",
    color_name="orange-70",
)


st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(msg["content"])


# "can doxycycline cause severe side effects in patients with acne?"

if question := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": question})

    st.chat_message("user").write(question)

    with st.spinner("Loading..."):

        context = search(question)

        # SYSTEM_PROMPT = """
        # You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided. If the context does not have sufficient information, you can use your own knowledge to provide a helpful answer.
        # """

        # USER_PROMPT = f"""
        # Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags. If the context does not provide enough information, rely on your own knowledge to answer.
        # <context>
        # {context}
        # </context>
        # <question>
        # {question}
        # </question>
        # """

        SYSTEM_PROMPT = """
        You are an AI assistant. You are able to find answers to the questions. Try using contextual passage snippets first for accurate and helpful answers. If the context does not have sufficient information, use your internal knowledge to provide a response that's as accurate and informative as possible.
        """
        USER_PROMPT = f"""
        Answer the following question enclosed in <question> tags. Try to prioritize using the following pieces of information enclosed in <context> tags to provide an answer to the question.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
        
        # SYSTEM_PROMPT = """
        # You are an AI assistant. Use your own knowledge if the context provided does not have sufficient information. However, try using contextual passage snippets first for accurate and helpful answers. If you're unsure about external sources, use your internal knowledge to provide a response that's as accurate and informative as possible.
        # """

        # USER_PROMPT = f"""
        # Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags. If the context does not provide enough information, rely on your own knowledge to answer. You can use external sources if necessary but be aware that using external knowledge might lead to less accurate responses.
        # <context>
        # {context}
        # </context>
        # <question>
        # {question}
        # </question>
        # """

        messages = [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            ("human", USER_PROMPT),
        ]
        response = llm.invoke(messages)

        if response is not None:

            st.spinner(False)

    print(response.content)

    st.session_state.messages.append(response.content)

    st.chat_message("assistant").write(response.content)
