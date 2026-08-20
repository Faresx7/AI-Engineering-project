from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import model as m
from pathlib import Path


class Retrieval:

    def __init__(self,
                embedding_model_name = "all-MiniLM-L6-v2",
                persist_dir = Path(__file__).resolve().parent / "db" / "chroma_db",
                k_chunks = 3
                 ):
    
      print("🔄 Initializing RAG Service & Loading Embeddings...")
      self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
      self.db = Chroma(persist_directory= str(persist_dir),
                      embedding_function= self.embedding_model,
                      collection_metadata={"hnsw:space":"cosine"}
                      )
      
      self.retriever = self.db.as_retriever(
                                  search_kwargs = {"k":k_chunks}     # retrieve the top 3 chunks
                                      )
      print('Rag system is Ready!✅')


    def build_rag_prompt(self, prompt):

        relevant_docs = self.retriever.invoke(prompt)
         
        context_text = "\n\n---\n\n".join([
                 f"[Document {i+1}]:\n{doc.page_content}"
                 for i, doc in enumerate(relevant_docs)
                 ])
         
        full_prompt = f"""You are a precise AI assistant. Answer the user's question based ONLY on the provided context below.    
### Rules:
1. Rely STRICTLY on the information in the provided context. Do NOT use prior knowledge or extrapolate beyond what is explicitly stated.
2. If the context does not contain enough information to answer the question accurately, respond ONLY with: "I don't know."
3. Do not make assumptions or try to guess.         
### Context:
{context_text}         
### User Question:
{prompt}
"""
         
        return full_prompt
