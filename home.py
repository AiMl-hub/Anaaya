import streamlit as st

from streamlit_extras.let_it_rain import rain

from streamlit_extras.colored_header import colored_header

from streamlit_card import card

from annotated_text import annotated_text, annotation

from PIL import Image

from langchain_together import ChatTogether
from openai import OpenAI

import os

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['TOGETHER_API_KEY'] = st.secrets['TOGETHER_API_KEY']

from pymilvus import MilvusClient
from tqdm import tqdm

openai_client = OpenAI()

EMBEDDING_DIM = 768

text_file = open("contexts.txt", "r")
text_content = text_file.read()
text_file.close()

docs = text_content.split("\n ")
docs = docs[:10]


llm = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding[:EMBEDDING_DIM]
    )


milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=EMBEDDING_DIM,  # Dimension of the embeddings",
    metric_type="COSINE",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)


data = []

for i, line in enumerate(tqdm(docs, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)


def search(question):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    return context


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


# "can doxycycline cause side effects in patients with acne?"

if question := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": question})

    st.chat_message("user").write(question)

    with st.spinner("Loading..."):

        context = search(question)

        SYSTEM_PROMPT = """
        You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """
        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

        print(context)

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
    