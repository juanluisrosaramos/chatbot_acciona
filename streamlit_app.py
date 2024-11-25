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
    from langchain.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""  # Concise and informative

    template = """Eres el CIO de Acciona.  Adopta un tono serio y formal,  como si te dirigieras a los accionistas de la compañía.  Tu objetivo es proporcionar respuestas claras,  extensas y con mucha información relevante para inversores.  Utiliza emojis con moderación para enfatizar puntos clave.

        Contexto: {context}

        Pregunta: {question}

        Respuesta (como CIO de Acciona): 💼

        (Aquí debes responder a la pregunta utilizando la información del contexto.  Sé preciso,  detallado y ofrece ejemplos concretos.  Recuerda que te diriges a inversores,  por lo que la información financiera y estratégica es crucial.  Tu respuesta debe ser fácilmente comprensible y dejar completamente clara la postura de Acciona.)

        Para profundizar en este tema,  sugiero las siguientes preguntas adicionales:

        1. suggestion1  🤔
        2. suggestion2  📊
        3. suggestion3  🚀
        """

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
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
    
    def process_query(result):
        
        answer = result["result"]
        sources = result["source_documents"]

        # Clean and format sources, removing the file path prefix
        source_strings = []
        seen_sources = set()
        file_path_prefix = "/content/drive/MyDrive/acciona/"  # Prefix to remove

        for doc in sources:
            source_str = f"{doc.metadata['source']} page #: {doc.metadata['page']}"
            if source_str not in seen_sources:
                # Remove the prefix if it exists
                cleaned_source = source_str.replace(file_path_prefix, "")
                source_strings.append(cleaned_source)
                seen_sources.add(source_str) # Still use original for duplicate check

        final_response = f"{answer}\n\nSources:\n{chr(10).join(source_strings[:3])}"
        return final_response


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  #  Or "map_reduce" if appropriate
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}  # Use the prompt template
    )
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
                #print(response)
                stream = process_query(response)

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            #response = st.markdown(stream)
            st.markdown(stream)
        st.session_state.messages.append({"role": "assistant", "content": response['result']})
