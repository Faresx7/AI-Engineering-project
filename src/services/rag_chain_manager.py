from src.core import container
from deep_translator import GoogleTranslator
from langdetect import detect

def generate_answer(text: str):
    rag_chain = container.rag_chain_instance
    if rag_chain is None:
        raise RuntimeError("RAG chain is not initialized")

    lang = detect(text)
    if lang is not "en":
        
        pass

    
    # Remove 'await' here
    return rag_chain.answer(text)

    from deep_translator import GoogleTranslator


def translate_to_english(text: str) -> str:
    """Translates incoming user query to English for better BM25/Vector retrieval."""
    try:
        # Automatically detects the language and converts to English
        translated = GoogleTranslator(source="auto", target="en").translate(
            text
        )
        return translated if translated else text
    except Exception as e:
        # Fallback to original text if translation service fails
        print(f"⚠️ Translation failed: {e}")
        return text