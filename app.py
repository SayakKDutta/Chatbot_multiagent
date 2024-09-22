import streamlit as st
from openai import OpenAI
st.set_page_config(layout="wide")
st.title("Emplochat")
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pprint
import os


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

###Enivironment settings for openai API key and Vector Embeddings############



client = OpenAI(api_key=API_KEY)   
persist_directory = '/mount/src/Chatbot_multiagent/embeddings'

#########################Loading the Stored Vector embeddings################
#Initialize the Chroma DB client
store = Chroma(persist_directory=persist_directory,collection_name="Capgemini_policy_embeddings")

# Get all embeddings
embeddings = store.get(include=['embeddings'])
embed_prompt = OpenAIEmbeddings(API_KEY)

###############################################################################

# Define the embedding retrieval function
def retrieve_vector_db(query, n_results=2):
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(
        embedding=embed_prompt.embed_query(query), k=n_results)
    results = []
    prev_embedding = []
    for embedding in similar_embeddings:
        if embedding not in prev_embedding:
            results.append(embedding)
        prev_embedding = embedding
    return results

# Initialize session states for storing messages
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query here?"):
    retrieved_results = retrieve_vector_db(query, n_results=3)
    context = ''.join([doc[0].page_content for doc in retrieved_results[:2]])

    # Determine the specialized head based on the query content
    if "leave" in query.lower():
        head = "Leave Policy Expert"
    elif "ethics" in query.lower():
        head = "Business Ethics Expert"
    elif "human rights" in query.lower():
        head = "Human Rights Expert"
    else:
        head = "General Policy Expert"

    # Construct the RAG prompt with context
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

    # Generate Normal RAG response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            max_tokens=1500,
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": prompt}],
            stream=True
        )
        normal_response = st.write_stream(stream)

    # Append the assistant's Normal RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": normal_response})

    # Check for vagueness
    def check_vagueness(answer):
        vague_phrases = ["I am not sure", "it depends", "vague", "uncertain", "unclear"]
        return any(phrase in answer.lower() for phrase in vague_phrases)

    is_vague_normal = check_vagueness(normal_response)

    # Score the response
    def calculate_relevance_score(query, response):
        keywords = query.lower().split()
        matches = sum(1 for word in keywords if word in response.lower())
        return matches / len(keywords)

    relevance_score_normal = calculate_relevance_score(query, normal_response)

    # Display Normal RAG vagueness and score metrics
    st.markdown(f"**Normal RAG Vagueness Detected:** {'Yes' if is_vague_normal else 'No'}")
    st.markdown(f"**Normal RAG Relevance Score:** {relevance_score_normal:.2f}")

    # Generate Multi-Agent RAG response
    with st.chat_message("assistant"):
        multi_prompt = f'''
        [INST]
        You are an expert in {head}. Provide a detailed response based on the context and your training.

        Question: {query}

        Context : {context}
        [/INST]
        '''
        stream_multi = client.chat.completions.create(
            max_tokens=1500,
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": multi_prompt}],
            stream=True
        )
        multi_response = st.write_stream(stream_multi)

    # Append the assistant's Multi-Agent RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": multi_response})

    # Check for vagueness in Multi-Agent response
    is_vague_multi = check_vagueness(multi_response)
    relevance_score_multi = calculate_relevance_score(query, multi_response)

    # Display Multi-Agent RAG vagueness and score metrics
    st.markdown(f"**Multi-Agent RAG Vagueness Detected:** {'Yes' if is_vague_multi else 'No'}")
    st.markdown(f"**Multi-Agent RAG Relevance Score:** {relevance_score_multi:.2f}")

# End of code

