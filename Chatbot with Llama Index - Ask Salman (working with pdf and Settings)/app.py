import os
import streamlit as st
#from llama_index import  ServiceContext
#from llama_index.llms import OpenAI
from llama_index.core import VectorStoreIndex
#from llama_hub.assemblyai.base import AssemblyAIAudioTranscriptReader 
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.title("💬 Ask Matilda")
   
def main():
    st.header("Ask Matilda")

    Settings.llm = OpenAI(model="gpt-4-turbo", temperature=0.5, system_prompt="You are an assistant and your job is to develop a question paper based on the input provided by the user.") 
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]   
    #reader = AssemblyAIAudioTranscriptReader(file_path="./audio.mp3")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    index  = VectorStoreIndex.from_documents(docs)
    query=st.text_input("Ask Matilda to make a question paper for you")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)
if __name__=='__main__':
    main()    
