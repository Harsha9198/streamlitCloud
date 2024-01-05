import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Documentation Assistant", page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.header("Chat with the CanvasToCode docs💬")
st.subheader("Talk to our canvas")
st.warning("Get Back To site 👉 [CanvasTocode](https://tic.comakeit.com/)", icon="🌐")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "🤖", "content": "Ask me a question on how to get started with CanvasToCode!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.7,system_prompt="You are a assistant for CodeToCanvas documentation and answer only when user gives a prompt, do not get queries of your own and donot give imaginary answers give only facts."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine( chat_mode="condense_question", verbose=True)
prompt=st.chat_input("Your question")
if prompt:# Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "🧑🏻‍💻", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("🤖"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "🤖", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history