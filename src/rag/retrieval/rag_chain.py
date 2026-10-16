import asyncio
import src.rag.retrieval_pipeline as retriever
import src.rag.models.cloud_model as model

class RAGChain:
    def __init__(self, k_chunks: int = 3):
        print("Getting ready...")
        
        self.retrieval = retriever.Retrieval(k_chunks = k_chunks)
        self.model = model.AIModel()
        
        print("✅[RAG_Chain] Model and retrieval is ready")


    async def answer(self, prompt: str):
        if not prompt.strip():
            return {
                "answer": "Please provide a valid Question.",
                "context": [],
                "final_prompt": "",
            }


        final_prompt, retrieved_content = await asyncio.to_thread(
            self.retrieval.build_rag_prompt, prompt
        )

        answer_text = await self.model.generate_response(final_prompt)

        return {
            "answer": answer_text,
            "context": retrieved_content,
            "final_prompt": final_prompt,
        }
    

# rc = RAGChain()
# while True:
#     i = input("enter you question here\n")
#     if i == 'e':
#         break
#     answer = rc.answer(i)
#     print(f"answer:\n{answer['answer']}")
  