from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import pickle

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("punkt", quiet= True)
nltk.download("stopwords", quiet= True)



class Retrieval:

    STOP_WORDS = set(stopwords.words("english"))
    PUNCTUATION_SET = set(string.punctuation)
    def __init__(self,
                embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                persist_dir = Path(__file__).resolve().parent / "storage" / "db" /"chroma_db",
                bm25_dir = Path(__file__).resolve().parent / "storage" / "bm25_index.pkl",
                vector_threshold = .2,
                k_chunks = 3
                 ):
        """
        Initializes the Retrieval system with embeddings, vector database, and BM25 index.
        
        Args:
            embedding_model_name (str): Name of the HuggingFace embedding model to use.
            bm25_dir (Path): Directory path to the BM25 index pickle file.
            persist_dir (Path): Directory path to the Chroma vector database.
            k_chunks (int): Number of top chunks to retrieve.
        
        Outputs:
            Initializes instance attributes: embedding_model, db, retriever, bm25, raw_text, and k_chunks.
        """
        
        print("🔄 Initializing RAG Service & Loading Embeddings...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.db = Chroma(persist_directory= str(persist_dir),
                        embedding_function= self.embedding_model,
                        collection_metadata={"hnsw:space":"cosine"}
                        )

        self.k_chunks = k_chunks
        self.retriever = self.db.as_retriever(
                                                search_type = "similarity_score_threshold",
                                                search_kwargs = {"k":k_chunks,
                                                                "score_threshold":vector_threshold},     # retrieve the top 3 chunks
                                                )

        self.bm25_dir = bm25_dir

        if self.bm25_dir.exists():
            with open(self.bm25_dir, "rb") as file:
                bm25_data = pickle.load(file)
                self.raw_text = bm25_data['raw_text']
                self.bm25 = bm25_data['bm25']
        else:
            self.bm25 = None
            self.raw_text = []
            print("⚠️ Warning: BM25 index file not found!")

        print('Rag system is Ready!✅')

    @classmethod
    def tokenize_and_remove_stopwords(cls, text: str) -> list[str]:
        """Tokenizes text, removes punctuation and stopwords using NLTK."""

        text = text.lower()
        tokens = word_tokenize(text)


        cleaned_tokens = [
            word
            for word in tokens
            if word not in cls.STOP_WORDS and word not in cls.PUNCTUATION_SET
        ]

        return cleaned_tokens


    def _bm25_index_retriever(self, prompt: str) -> list[str]:
        """Retrieves top matching documents using BM25 with score filtering,
            retrieve the k_chunks that have been defined in __init__()."""
        if not self.bm25:
            return []

        tokenized_prompt = self.tokenize_and_remove_stopwords(prompt)
        scores = self.bm25.get_scores(tokenized_prompt)

        # Pair each document text with its score
        doc_score_pairs = list(zip(self.raw_text, scores))

        # Filter out documents with zero score
        matching_docs = [(doc, score) for doc, score in doc_score_pairs if score > 0]

        if not matching_docs:
            return []

        # Sort descending by score
        matching_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in matching_docs[: self.k_chunks]]


    def _vector_index_retriever(self, prompt) -> list[str]:
        """Retrieves top matching documents using vector similarity search,
            retrieve the k_chunks that have been defined in __init__().
        
        Args:
            prompt (str): The query prompt to retrieve relevant documents for.
        
        Returns:
            list[str]: List of top k_chunks documents sorted by similarity score.
        """

        relevant_docs = self.retriever.invoke(prompt)
        retrieved_docs = [doc.page_content for doc in relevant_docs]
        return retrieved_docs[:self.k_chunks]


    def _combine_retrieved_docs(self, prompt):
        """Combines and deduplicates documents from BM25 and vector retrievers,
        
        
        Args:
            prompt (str): The query prompt to retrieve relevant documents for.
        
        Returns:
            list[str]: List of combined unique documents from both BM25 and vector search,
                      limited to k_chunks*2 documents.
        """

        bm25_docs = self._bm25_index_retriever(prompt)
        vector_docs = self._vector_index_retriever(prompt)

        combined_docs = []
        seen = set()
        max_len = max(len(vector_docs), len(bm25_docs))
    
        for i in range(max_len):
            if i < len(vector_docs) and vector_docs[i] not in seen:
                combined_docs.append(vector_docs[i])
                seen.add(vector_docs[i])
            if i < len(bm25_docs) and bm25_docs[i] not in seen:
                combined_docs.append(bm25_docs[i])
                seen.add(bm25_docs[i])

        return combined_docs[: self.k_chunks]


    def build_rag_prompt(self, prompt):
        """Builds a RAG prompt by combining retrieved context with user query and instructions.
        
        Args:
            prompt (str): The user's question to retrieve relevant documents for.
        
        Returns:
            str: A formatted prompt containing system instructions, context from retrieved documents,
                 and the user's original question, ready for LLM processing.
        """
        
        retrieved_chunks = self._combine_retrieved_docs(prompt)
        context_text = "\n---\n".join(retrieved_chunks)
         
        full_prompt = f"""
You are a precise AI assistant. Answer the user's question accurately based ONLY on the provided context below.

### Rules for Processing Context:
1. **Handle Multiple Listings & Conflicts:** If a product appears multiple times with different status (e.g., one listing has warranty, another does not), explicitly mention all available variations/options to the user.
2. **Apply General Rules to Specific Models:** Use numerical/version logic (e.g., iPhone 11 is older than iPhone 16) to correctly apply general warranty duration rules to the specific model asked about.
3. **Clean Output:** NEVER mention "Document 1", "Document 2", or any context metadata in your final response. Answer naturally.
4. **Strict Grounding:** Rely ONLY on the facts explicitly stated. Do NOT invent new details.
5. **Insufficient Info:** If the context lacks sufficient information, reply ONLY with: "I don't know."

### Context:
{context_text}

### User Question:
{prompt}
"""
         
        return full_prompt