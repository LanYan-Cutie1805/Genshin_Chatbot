#IMPORITNG REQUIRED LIBRARY
import streamlit as st
import emoji
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
import os
import json
import pandas as pd
import re
from PIL import Image

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)


#SETTING UP LOGGING
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


#DEFINING THE SYSTEM PROMPT
system_prompt = """
You are a knowledgeable Genshin Impact assistant. You have access to a detailed 
database of Genshin Impact characters, including their elements, weapons, skills, 
ascension materials, and roles. 

Your job is to help users by answering questions about Genshin Impact characters 
based on the provided data. If a user asks about a character’s best weapons, 
artifacts, or team compositions, use the data available in the CSV file. 
If you don’t know the answer, say that you DON'T KNOW.

Answer all in English.
Your task is to be a friendly and knowledgeable assistant in the world of Teyvat.
You will help Travelers by providing information about characters, elements, 
weapons, artifacts, and their roles in combat.

You have knowledge of all released characters in Genshin Impact and can give 
recommendations based on user needs.

Your Responsibilities:
Answer Travelers' questions about characters, elements, weapons, and the best builds. 
Help Travelers understand character roles, such as main DPS, sub-DPS, or support. 
Provide accurate information based on the available database.

Rules:
Only provide information based on available data. If you cannot find a relevant answer, say that you don’t know.
Ensure each response is clear, accurate, and easy for Travelers to understand.
Do not make assumptions or provide information outside the scope of Genshin Impact.

Conversation so far:
"""

#COFIGURE THE LLM
Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


#LOADING THE CSV DATA
@st.cache_resource(show_spinner="Loading character data – please be patient.")
def load_data(vector_store=None):
    with st.spinner(text="Preparing character data – please be patient."):
        csv_parser = CSVReader(concat_rows=False)

        #Read PDF file
        pdf_parser = PyMuPDFReader()
        
        file_extractor = {".csv": csv_parser, ".pdf": pdf_parser}

        # Read & load document from folder
        reader = SimpleDirectoryReader(
            input_dir="./docs",
            recursive=True,
            file_extractor=file_extractor,

            # Suppress file metadata, not sure if this works or not.
            file_metadata=lambda x: {}
        )
        documents = reader.load_data()

    
    #CREATING SEARCHABLE DATABASE (VECTOR INDEX AND BM25)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index_retriever = index.as_retriever(similarity_top_k=8)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=16,
    )


    return QueryFusionRetriever(
        [index_retriever, bm25_retriever],
        num_queries=2,
        use_async=True,
        similarity_top_k=24
    )


# Function to get image path from the CSV file
df = pd.read_csv("./docs/merged_character_data(2).csv")  # Load CSV globally
def show_character_image(character_name):
    """Retrieve and display a character's image based on user input."""
    row = df[df["Name"].str.lower() == character_name.lower()]  # Ensure correct column name

    if not row.empty:
        image_path = row.iloc[0]["Image"].strip()  # Get image path from CSV
        abs_path = os.path.abspath(image_path)  # Convert to absolute path
        if os.path.exists(abs_path):
            return abs_path  # ✅ Correct: Returning the absolute path
        else:
            st.error(f"❌ Image not found at: {abs_path}")
            return None
    else:
        st.error(f"⚠️ No data found for {character_name}")
        return None



#SETTING UP THE CHATBOT UI
# Main Program
col1, col2, = st.columns([1, 1.4])

with col1:
    st.title("Hello Traveller")
with col2:
    st.image("paimon.png", width=80)
st.write("Welcome to the Genshin Impact Chatbot ✨")
retriever = load_data()

#MANAGING CHAT HISTORY
# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "What can I help? Searching for character stats, compare characters? 😉"}
    ]

#INITIALIZE THE CHAT ENGINE
# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="What can I help? Searching for character stats, compare characters? 😉"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    st.session_state.chat_engine = CondensePlusContextChatEngine(
        verbose=True,
        system_prompt=system_prompt,
        context_prompt=(
                "You are a Genshin Impact assistant who helps players find character information.\n"
                "Format of the database: Character, Element, Weapon, Role, Skills, Best Artifacts, Best Weapons\n"
                "Here is relevant character data:\n\n"
                "{context_str}"
                "\n\nUse this context to help the user with their question."
            ),
        condense_prompt="""
Given a conversation (between the User and Assistant) and a follow-up message from the User,
Rewrite the follow-up message as an independent question that includes all relevant context from the previous conversation. The independent question should be a single sentence. Important details include the character’s name, role, abilities, weapon, element, or any other relevant game-related information.

Example of a standalone question:
"What are the best artifact sets for Hu Tao in a Vaporize team?"

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>""",
        memory=memory,
        retriever=retriever,
        llm=Settings.llm
    )



if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None
if "last_character" not in st.session_state:
    st.session_state.last_character = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(message["content"], caption=message.get("character_name", "image"), use_column_width=True)
        else:
            st.markdown(message["content"])


        
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if user asks for an image
    trigger_words = ["show", "display", "generate", "find", "give me", "photo", "picture"]
    image_words = ["image", "picture", "photo"]
    if st.session_state.last_image_path:
        st.image(st.session_state.last_image_path, caption=st.session_state.last_character, use_column_width=True)
    if any(word in prompt.lower() for word in trigger_words) and any(img_word in prompt.lower() for img_word in image_words):
        cleaned_prompt = prompt.lower()
        for word in trigger_words + image_words:
            cleaned_prompt = cleaned_prompt.replace(word, "")


        cleaned_prompt = re.sub(r"\b(of|a|an|the|me|)\b", "", cleaned_prompt).strip()
        character_name = cleaned_prompt.title()  # Convert to title case (Keqing, Zhongli, etc.)
        if character_name:
            image_path = show_character_image(character_name)
            if image_path:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "image",
                    "content": str(image_path),
                    "character_name": character_name
                })
                with st.chat_message("assistant"):
                    st.image(image_path, caption=character_name, use_column_width=True)
            else:
                st.error("❌ Image path is None. Something went wrong.")
        else:
            st.error("⚠️ Character name not recognized.")
    
    else:
        # Process normal text-based chatbot responses
        with st.chat_message("assistant"): 
            placeholder = st.empty()    
            with st.spinner("Paimon is thinking..."):
                placeholder.image("paimon-think.jpg", width=200)
                
                response_stream = st.session_state.chat_engine.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)
            placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
