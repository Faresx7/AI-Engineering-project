from src.core import container

def answer(text: str):
    rag_chain = container.rag_chain_instance
    if rag_chain is None:
        raise RuntimeError("RAG chain is not initialized")
    
    # Remove 'await' here
    return rag_chain.answer(text)