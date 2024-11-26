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

# Memory and Chat History
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Set up memory (same key as in the prompt template)
msgs = StreamlitChatMessageHistory(key="chat_history")  # Consistent key
if len(msgs.messages) == 0:
    msgs.add_ai_message("¡Hola! Soy el CFO de Acciona. ¿En qué puedo ayudarte?")
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

        # Set up memory (same key as in the prompt template)
    msgs = StreamlitChatMessageHistory(key="chat_history")  # Consistent key
    if len(msgs.messages) == 0:
        msgs.add_ai_message("¡Hola! Soy el bot de ayuda al accionista de Acciona. ¿En qué puedo ayudarte?")
    template = """Eres un financiero de Acciona Energía tienes acceso a los ESTADOS FINANCIEROS SEMESTRALES RESUMIDOS CONSOLIDADOS E
        INFORME DE GESTIÓN INTERMEDIO CORRESPONDIENTES AL PERIODO DE SEIS MESES TERMINADO A 30 DE JUNIO DE 2024. 
        
        Adopta un tono serio y formal, como si te dirigieras a los accionistas de la compañía. 
        Tu objetivo es proporcionar respuestas claras, extensas y con mucha información relevante para inversores. 
        Utiliza emojis con moderación para enfatizar puntos clave.  
        
        Historial de la conversación:
        {chat_history}

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta: 💼

        Debes responder a la pregunta utilizando la información del contexto. Sé preciso, detallado y ofrece ejemplos concretos. 
         - Recuerda que te diriges a inversores, por lo que la información financiera y estratégica es crucial.
         - Es crucial que no añadas hechos e información que no estén en el contexto. 
         - Cita las fuentes relevantes usando el formato [número] con enlace al documento a la página.  
        
        Por ejemplo:  "[1](enlace_al_documento_1#página_1)". 
        
        Formatea tu respuesta, por ejemplo:

        * **Negrita** para enfatizar.
        * *Cursiva* para nombres de documentos o términos técnicos.
        * Listas con viñetas o numeradas para organizar la información.
        * Enlaces clicables para las fuentes: [texto del enlace](URL).
        * Saltos de línea para párrafos separados.

        Tu respuesta debe ser fácilmente comprensible y dejar completamente clara la postura de Acciona Energía.
        """
    PROMPT = PromptTemplate(
        input_variables=["context", "question", "chat_history"],  # Add chat_history
        template=template,)       
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
        search_kwargs={"k": 8, "score_threshold": 0.7},
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
    
    from langchain_openai import ChatOpenAI

    chain = PROMPT | ChatOpenAI(api_key=openai_api_key, model='gpt-4o', temperature=0)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
        )
  
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
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
    
    if prompt := st.chat_input("Pregunta al financiero de Acciona Energía:"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            with st.spinner("Pensando..."):

                docs = retriever.get_relevant_documents(prompt)
                # Construct the context string
                #context = "\n".join([doc.page_content for doc in docs])
                context = "\n".join([f"**Source {i+1}:** *{doc.metadata['source'].replace('/content/drive/MyDrive/acciona/', '')} page #: {doc.metadata['page']}*\n{doc.page_content}" for i, doc in enumerate(docs[:3])])
                config = {"configurable": {"session_id": "any"}} # Necessary for history to work
                inputs = {
                    "question": prompt,
                    "chat_history": msgs.messages,
                    "context": context,  # Include the retrieved context
                }
                print(context)
                
                response = chain_with_history.invoke(inputs, config)
                
                print(response)
                #stream = process_query(response)

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            st.markdown(response.content)

