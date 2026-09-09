import retrieval_pipeline as retriever
import model as model

class RAGChain:
    def __init__(self, k_chunks: int = 3):
        print("Getting ready...")
        self.retrieval = retriever.Retrieval(k_chunks = k_chunks)
        self.model = model.AIModel()
        print("✅ Model and retrieval is ready")


    def answer(self, prompt: str):
        if not prompt.strip():
            return {"answer": "Please provide a valid Question.", "context": []}


        final_prompt, retrieved_content = self.retrieval.build_rag_prompt(prompt)
        
        answer = self.model.generate_response(final_prompt)
        
        return {"answer":answer,
                "context":retrieved_content,
                "final_prompt": final_prompt}
