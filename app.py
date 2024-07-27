import streamlit as st
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import ollama

# Initialize the embeddings models
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant clients
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="chatbot")

# Streamlit app
st.title("Llama 3 Contextual Recommendation System")
st.image(r"C:\Users\srika\Downloads\bg_rec.png"", use_column_width=True)  # Add your bg_rec.png as the logo")

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    page = st.selectbox(
        "Navigation", 
        ["Home", "Generate Response", "Contact Us"],
        format_func=lambda x: "Home" if x == "Home" else "Generate Response" if x == "Generate Response" else "Contact Us"
    )

if page == "Home":
    st.subheader("Home")
    st.success("Retrieval-Augmented Generation (RAG) using the Hugging Face Open Source Embedding Model integrated with LLAMA3. Made by A SRI KARTHIK, NIKITHA A R, AKSHITHA.")
    st.write("Welcome to the Chat application. Select 'Generate Response' from the menu to get started.")
    st.write("Using cutting-edge AI technology, It is an independent LLM-based Query agent. By utilizing cutting-edge models like the Open Source Model from Hugging Face and the BGE-Large-EN embeddings from BAAI, BookFinder provides an advanced retrieval-augmented generation (RAG) capacity. Through a conversational chat interface, users may interact and receive tailored suggestions based on their inquiries. Semantic search and deep learning models are easily integrated by the application to deliver precise and contextually relevant results. It was created with the user in mind, combining the effectiveness of AI-driven recommendation algorithms with natural user interfaces to improve the experience.")

elif page == "Generate Response":
    st.subheader("Chat with Llama")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    # Display all messages
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.text_area(f"User-{index}", value=message["content"], height=50, disabled=True)
        elif message["role"] == "assistant":
            st.text_area(f"Assistant-{index}", value=message["content"], height=50, disabled=True)

    # Chat input
    prompt = st.text_input("Say something")

    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Perform a semantic search
            docs = db.similarity_search_with_score(query=prompt, k=5)

            # Prepare the context for the Llama3 model
            context = "\n".join([doc.page_content for doc, score in docs])

            # Generate a response using Llama3
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': f"You are a recommendation system, Using the following context, write about {prompt}:\n{context}",
                },
            ])

            # Build full response
            full_response = response['message']['content']

        # Display assistant's response with a unique key
        st.text_area(f"Assistant-{len(st.session_state.messages)}", value=full_response, height=50, disabled=True)
        
        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif page == "Contact Us":
    st.subheader("Contact Me")
    st.write("Please fill out the form below to get in touch with me.")

    # Input fields for user's name, email, and message
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message", height=150)

    # Submit button
    if st.button("Submit"):
        if name.strip() == "" or email.strip() == "" or message.strip() == "":
            st.warning("Please fill out all the fields.")
        else:
            send_email_to = 'srikarthikavala@gmail.com'
            st.success("Your message has been sent successfully!")

st.sidebar.success("This app demonstrates Retrieval-Augmented Generation (RAG) using the Llama3 model.")
st.sidebar.warning("Recommnder Systems [J COMP]")
