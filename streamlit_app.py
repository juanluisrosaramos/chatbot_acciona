import streamlit as st
from openai import OpenAI

import os
import openai, langchain, pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from pinecone import Pinecone
import os
from pinecone import Pinecone # import the Pinecone class
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from pinecone import ServerlessSpec

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot For acciona documents "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "    
)




# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key  = None
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")    
else:
    
    OPENAI_API_KEY = openai_api_key

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    embed_model = "text-embedding-ada-002"
    os.environ["PINECONE_API_KEY"] = "pcsk_2fqiNg_Bk993SPWiRm9zedaJcD71eBfr43rsNzmtBhaAS2RPyRJZ15m3CPRVSLNPQDvTig"
    PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
    PINECONE_ENV = "us-west4-gcp-free"
    PINECONE_ENV = "us-west4-gcp-free"
    index_name = 'accionapdfs'
    api_key = PINECONE_API_KEY

    # configure client
    pc = Pinecone(api_key=api_key)


    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'

    spec = ServerlessSpec(cloud=cloud, region=region)

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
    # connect to index
    index = pc.Index(index_name)
    # view index stats
    index.describe_index_stats()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.7},
    )
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    def parse_response(response):
        print(response['result'])
        print('\n\nSources:')
        for source_name in response["source_documents"]:
            print(source_name.metadata['source'], "page #:", source_name.metadata['page'])
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True)
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("¿Qué puede hacer acciona por los accionistas?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            with st.spinner("Pensando..."):
                response = qa_chain(prompt)
                print(response)
                stream = parse_response(response)
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.markdown(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
