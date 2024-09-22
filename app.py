import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain_community.vectorstores import Chroma # Updated import
import os
import time  # For simulating delays

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("Emplochat")

# Sidebar for API Key input
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# Define the embedding function
class OpenAIEmbeddingFunction:
    def __call__(self, texts):
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [embedding['embedding'] for embedding in response['data']]
        return embeddings

# Initialize OpenAI client
class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def chat(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

# Initialize the OpenAI client
client = OpenAIClient(API_KEY)
persist_directory = '/mount/src/Chatbot_multiagent/embeddings'

# Initialize the Chroma DB client
store = Chroma(persist_directory=persist_directory, collection_name="Capgemini_policy_embeddings")

# Retrieve embeddings and initialize other components
embed_prompt = OpenAIEmbeddingFunction()

# Function to retrieve embeddings
def retrieve_vector_db(query, n_results=2):
    embedding_vector = embed_prompt([query])[0]
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding=embedding_vector, k=n_results)
    results = []
    prev_embedding = []
    for embedding in similar_embeddings:
        if embedding not in prev_embedding:
            results.append(embedding)
        prev_embedding = embedding
    return results

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query here?"):
    retrieved_results = retrieve_vector_db(query, n_results=3)
    context = ''.join([doc[0].page_content for doc in retrieved_results[:2]])

    # Determine specialized head
    if "leave" in query.lower():
        head = "Leave Policy Expert"
    elif "ethics" in query.lower():
        head = "Business Ethics Expert"
    elif "human rights" in query.lower():
        head = "Human Rights Expert"
    else:
        head = "General Policy Expert"

    # Construct the RAG prompt
    prompt = f'''
    [INST]
    You are an expert in {head}. Give a detailed answer based on the context provided and also your training.
    Question: {query}
    Context : {context}
    [/INST]
    '''

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display the user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Normal RAG response with a progress bar
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            progress = st.progress(0)
            normal_response = ""
            stream = client.chat(
                max_tokens=1500,
                model="gpt-3.5-turbo",  # Adjust model as necessary
                messages=[{"role": "system", "content": prompt}],
                stream=True
            )

            # Simulate response streaming
            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    normal_response += chunk['choices'][0]['delta']['content']
                    st.write(chunk['choices'][0]['delta']['content'], end="")
                time.sleep(0.1)  # Simulate delay (for demonstration)

            st.session_state.messages.append({"role": "assistant", "content": normal_response})

    # Vagueness and relevance score check
    is_vague_normal = check_vagueness(normal_response)
    relevance_score_normal = calculate_relevance_score(query, normal_response)

    # Display metrics
    st.markdown(f"**Normal RAG Vagueness Detected:** {'Yes' if is_vague_normal else 'No'}")
    st.markdown(f"**Normal RAG Relevance Score:** {relevance_score_normal:.2f}")
