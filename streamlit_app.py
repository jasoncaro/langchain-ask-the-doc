import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

# Retrieve the OpenAI API key from st.secrets
openai_api_key = st.secrets["openai_api_key"]

# Define summarization function
def generate_summary(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        text = uploaded_file.read().decode()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
        chunks = text_splitter.create_documents([text])

        map_custom_prompt = '''
        Summarize the following text in a clear and concise way:
        TEXT:`{text}`
        Brief Summary:
        '''

        combine_custom_prompt = '''
        Generate a summary of the following text that includes the following elements:

        * A title that accurately reflects the content of the text.
        * An introduction paragraph that provides an overview of the topic.
        * Bullet points that list the key points of the text.
        * A conclusion paragraph that summarizes the main points of the text.

        Text:`{text}`
        '''

        map_prompt_template = PromptTemplate(
            input_variables=['text'],
            template=map_custom_prompt
        )

        combine_prompt_template = PromptTemplate(
            input_variables=['text'],
            template=combine_custom_prompt
        )

        llm = OpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo')

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=False
        )

        summary = summary_chain.run(chunks)
        return summary

# Streamlit UI
st.set_page_config(page_title='Summarizer')
st.title('📝 Article Summarizer')

# File upload
uploaded_file = st.file_uploader('Upload a text file for summarization', type='txt')

# Button to trigger summarization
if st.button('Summarize') and uploaded_file:
    with st.spinner('Generating summary...'):
        summary_text = generate_summary(uploaded_file)
        st.subheader('Summary')
        st.write(summary_text)
else:
    st.write('Upload a file and click "Summarize" to start.')

