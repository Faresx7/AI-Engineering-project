import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader    # for reading files
from langchain_text_splitters import CharacterTextSplitter      # for chunking
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma     # for vector DB
from dotenv import load_dotenv      # To read sensitive data from .env files
import shutil

load_dotenv()


def load_documents(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f'Path "{path}" does not exist')

    loader = DirectoryLoader(
        path = path,
        glob = "*.txt",     # filters files only want to read
        loader_cls = TextLoader     # define the class that responsible for reading files inside the folder
        )
    
    docs = loader.load()

    if len(docs) == 0:
        raise FileNotFoundError(f'There is no files in "{path}" folder')

    return docs


def chunk_documents(docs, chunk_size = 650, chunk_overlap = 0, print_data = False):

    text_splitter = CharacterTextSplitter(chunk_size = chunk_size,
                                          chunk_overlap = chunk_overlap     # helps model keep the meaning
                                          )

    chunks = text_splitter.split_documents(docs)

    if print_data and chunks:
        for i, chunk in enumerate(chunks):
            print(f'\n--- chunk {i} ---')
            print(f'Source {chunk.metadata}')
            print(f'Length {len(chunk.page_content)} chars')
            print('Content')
            print(chunk.page_content)
            print('-'*50)

    return chunks


def create_vector_store(chunks, persist_dir = 'AI model/db/chroma_db'):
    """
    Embeds text chunks using HuggingFace model and stores them in Chroma DB.
    Model max context capacity: 256 tokens (~600-700 chars).
    """

    
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding = embedding_model,
        persist_directory = persist_dir,
        collection_metadata= {"hnsw:space":"cosine"}
     )
    
    return vector_store


def get_doc_info(doc):
    '''
    return additional informations about file
    '''
    return f'Letters: {len(doc.page_content)}, Words: {len(doc.page_content.split())}, MetaData: {doc.metadata}'


def main():

    files = load_documents('AI model/docs')
    chunks = chunk_documents(files,chunk_overlap=70)
    vs = create_vector_store(chunks)


if __name__ == "__main__":
    main()
