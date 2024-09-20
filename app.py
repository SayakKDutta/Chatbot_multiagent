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
embed_prompt = OpenAIEmbeddings()

###############################################################################




# Function to retrieve from the vector database
def retrieve_vector_db(query, n_results=3):
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(
        embedding=embed_prompt.embed_query(query), k=n_results
    )
    results = []
    prev_embedding = []

    for embedding in similar_embeddings:
        if embedding not in prev_embedding:
            results.append(embedding)
        prev_embedding = embedding

    return results

# Normal RAG response function
def normal_rag_response(query, temperature=0, max_tokens=200, top_n=10):
    retrieved_results = retrieve_vector_db(query, n_results=top_n)
    context = ''.join([doc[0].page_content for doc in retrieved_results[:2]])

    prompt = f'''
    [INST]
    Give an answer for the question based on the context provided and also on what you have been trained on.
    Question: {query}
    Context: {context}
    [/INST]
    '''

    completion = client.completions.create(
        temperature=temperature,
        max_tokens=max_tokens,
        model="ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj",
        messages=[
            {"role": "system", "content": "You are an expert in Capgemini policies."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Multi-head RAG response function
def multi_head_rag_response(query, temperature=0, max_tokens=200, top_n=10):
    if "leave" in query.lower():
        head = "Leave Policy Expert"
    elif "ethics" in query.lower():
        head = "Business Ethics Expert"
    elif "human rights" in query.lower():
        head = "Human Rights Expert"
    else:
        head = "General Policy Expert"

    retrieved_results = retrieve_vector_db(query, n_results=top_n)
    context = ''.join([doc[0].page_content for doc in retrieved_results[:2]])

    prompt = f'''
    [INST]
    You are an expert in {head}. Give a detailed answer based on the context provided and also your training.
    Question: {query}
    Context: {context}
    [/INST]
    '''

    completion = client.completions.create(
        temperature=temperature,
        max_tokens=max_tokens,
        model="ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj",
        messages=[
            {"role": "system", "content": f"You are an expert in {head}."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Function to check for vagueness
def check_vagueness(answer):
    vague_phrases = ["I am not sure", "it depends", "vague", "uncertain", "unclear"]
    return any(phrase in answer.lower() for phrase in vague_phrases)

# Retry if vague
def retry_if_vague(query, model_function, max_attempts=3):
    for attempt in range(max_attempts):
        answer = model_function(query)
        if not check_vagueness(answer):
            return answer
    return "Unable to generate a clear response."

# Calculate relevance score
def calculate_relevance_score(query, response):
    keywords = query.lower().split()
    matches = sum([1 for word in keywords if word in response.lower()])
    return matches / len(keywords)

# Calculate clarity score
def calculate_clarity(response):
    length = len(response.split())
    if 50 <= length <= 200:
        return 1.0
    elif length > 200:
        return 0.7
    else:
        return 0.5

# Function to log comparison of responses
def log_comparison(query, normal_rag_result, multi_head_result):
    normal_relevance = calculate_relevance_score(query, normal_rag_result)
    multi_head_relevance = calculate_relevance_score(query, multi_head_result)

    normal_is_vague = check_vagueness(normal_rag_result)
    multi_head_is_vague = check_vagueness(multi_head_result)

    normal_clarity = calculate_clarity(normal_rag_result)
    multi_head_clarity = calculate_clarity(multi_head_result)

    st.write(f"Query: {query}\n")

    st.write("Normal RAG Response:")
    st.write(f"Response: {normal_rag_result}")
    st.write(f"Relevance Score: {normal_relevance:.2f}")
    st.write(f"Vagueness Detected: {'Yes' if normal_is_vague else 'No'}")
    st.write(f"Clarity Score: {normal_clarity:.2f}\n")

    st.write("Multi-head Agentic RAG Response:")
    st.write(f"Response: {multi_head_result}")
    st.write(f"Relevance Score: {multi_head_relevance:.2f}")
    st.write(f"Vagueness Detected: {'Yes' if multi_head_is_vague else 'No'}")
    st.write(f"Clarity Score: {multi_head_clarity:.2f}\n")

    st.write("Comparison Result:")
    if normal_relevance > multi_head_relevance:
        st.write("Normal RAG had a better relevance score.")
    elif multi_head_relevance > normal_relevance:
        st.write("Multi-head Agentic RAG had a better relevance score.")
    else:
        st.write("Both models had equal relevance scores.")

    if normal_is_vague and not multi_head_is_vague:
        st.write("Multi-head Agentic RAG was less vague.")
    elif multi_head_is_vague and not normal_is_vague:
        st.write("Normal RAG was less vague.")

    if normal_clarity > multi_head_clarity:
        st.write("Normal RAG provided a clearer response.")
    elif multi_head_clarity > normal_clarity:
        st.write("Multi-head Agentic RAG provided a clearer response.")
    else:
        st.write("Both responses had similar clarity.")

# Streamlit app main function
st.title('Capgemini Policy Chatbot')

query = st.text_input("Enter your query:")
if st.button("Get Responses"):
    if query:
        normal_response = retry_if_vague(query, normal_rag_response)
        multi_head_response = retry_if_vague(query, multi_head_rag_response)

        st.write("Normal RAG Response:")
        st.write(normal_response)

        st.write("\nMulti-head Agentic RAG Response:")
        st.write(multi_head_response)

        if st.button("Evaluate Responses"):
            log_comparison(query, normal_response, multi_head_response)
    else:
        st.write("Please enter a query.")

