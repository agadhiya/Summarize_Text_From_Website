import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from PIL import Image

## Define your Groq API Key here
groq_api_key = "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" 

## Streamlit App

## #Title of the app
st.set_page_config(page_title="Somic: Summarize Text From Website")
st.title("Somic: Summarize Text From Website")
#st.markdown("<h1 style='color: red;'>Somic: Summarize Text From The Website</h1>", unsafe_allow_html=True)

# Open the image using Pillow
img = Image.open("logo.jpg")
img_resized = img.resize((700, 300))
# Display the company logo in the sidebar
st.sidebar.image(img_resized, use_container_width=True)   

## Get the URL to be summarized
generic_url = st.text_input("Enter URL:")

## Gemma Model Using Groq API
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 500 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Loading the website or YouTube video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                ## Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
