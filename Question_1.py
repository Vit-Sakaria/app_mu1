import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = {
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
}

# Check if any required environment variables are missing
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    st.error(
        "Missing required environment variables. Please set the following variables in your .env file:"
    )
    for var in missing_vars:
        st.error(f"- {var}")
    st.stop()

# Configure Streamlit page
st.set_page_config(page_title="AI MCQ Generator", page_icon="🎓", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🎓 AI-Powered MCQ Generator")
st.markdown(
    """
    Upload a PDF document related to Generative AI, and this tool will automatically generate:
    - 5 Multiple Choice Questions with options and answers
    - 5 Short Answer Questions
    """
)


def process_pdf(pdf_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Clean up temporary file
    os.unlink(tmp_path)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(pages)

    return chunks


def create_vector_store(chunks):
    try:
        # Get API version from environment variable
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable is not set")

        # Initialize Azure OpenAI embeddings with correct parameter names
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            chunk_size=1,
        )

        # Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.error("Please check your Azure OpenAI credentials in the .env file")
        return None


def generate_questions(vector_store):
    if vector_store is None:
        return None, None

    try:
        # Get API version from environment variable
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable is not set")

        # Initialize Azure OpenAI
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=api_version,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
        )

        # MCQ Generation Prompt
        mcq_prompt = PromptTemplate(
            template="""Based on the following context, generate 5 multiple choice questions with 4 options each and their correct answers.
            Format each question as:
            Q: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Answer: [Correct option letter]
            
            Context: {context}
            """,
            input_variables=["context"],
        )

        # Short Answer Question Prompt
        saq_prompt = PromptTemplate(
            template="""Based on the following context, generate 5 short answer questions.
            Format each question as:
            Q: [Question]
            Expected Answer: [Brief answer]
            
            Context: {context}
            """,
            input_variables=["context"],
        )

        # Create QA chains
        mcq_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": mcq_prompt},
        )

        saq_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": saq_prompt},
        )

        # Generate questions
        mcq_result = mcq_chain.invoke({"query": "Generate MCQs"})
        saq_result = saq_chain.invoke({"query": "Generate short answer questions"})

        return mcq_result["result"], saq_result["result"]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None, None


# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF and generating questions..."):
        # Process PDF
        chunks = process_pdf(uploaded_file)

        # Create vector store
        vector_store = create_vector_store(chunks)

        if vector_store is not None:
            # Generate questions
            mcqs, saqs = generate_questions(vector_store)

            if mcqs and saqs:
                # Display results
                st.markdown("## 📝 Generated Questions")

                # MCQs
                st.markdown("### Multiple Choice Questions")
                st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.markdown(mcqs)
                st.markdown("</div>", unsafe_allow_html=True)

                # Short Answer Questions
                st.markdown("### Short Answer Questions")
                st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.markdown(saqs)
                st.markdown("</div>", unsafe_allow_html=True)
