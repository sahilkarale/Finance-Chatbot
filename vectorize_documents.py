from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Loading Embedding model
embeddings = HuggingFaceEmbeddings()

# Loading the PDFs
loader = DirectoryLoader(
    path=r"4. Project\Data",  # Corrected path format
    glob="*.pdf",  # Corrected glob pattern
    loader_cls=UnstructuredFileLoader
)

documents = loader.load()

# Corrected parameter name from 'chunks_size' to 'chunk_size'
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents=documents)

# Creating VectorDB
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=r"4. Project\vector_db_dir"  # Corrected path format
)

print("Documents Vectorized")
