import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile
from gtts import gTTS
import os
import pygame

inference_api_key="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"

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_filename = temp_file.name
    
    tts.save(temp_filename)
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the audio file
    sound = pygame.mixer.Sound(temp_filename)
    
    # Play the audio file
    sound.play()
    
    # Wait for the duration of the audio file
    pygame.time.wait(int(sound.get_length() * 1000))  # Convert to milliseconds
    
    # Clean up - remove the temporary audio file
    os.remove(temp_filename)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vector_store(text_chunks):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(temperature=0,groq_api_key="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", model_name="llama3-8b-8192")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)  # Debugging line
    
    st.write(" Replies: ")
    if isinstance(response["output_text"], str):
        response_list = [response["output_text"]]
    else:
        response_list = response["output_text"]
    
    for text in response_list:
        st.write(text)
        # Convert text to speech for each response
        text_to_speech(text)

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")


    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                
if __name__ == "__main__":
    main()
