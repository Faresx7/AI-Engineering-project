import re
from deep_translator import MyMemoryTranslator
from src.core import container


async def generate_answer(text: str):
    """Generate response using RAG chain for incoming user query."""
    rag_chain = container.rag_chain_instance

    if rag_chain is None:
        raise RuntimeError("RAG chain is not initialized")

    if not text or not text.strip() or text == "[NON_TEXT_MESSAGE]":
        return ""

    english_text: str | None = None
    if has_arabic(text):
        english_text = translate_to_english(text)

    prompt = f"always answer in arabic {english_text or text}"
    response = await rag_chain.answer(prompt)

    return response["answer"]


def translate_to_english(text: str) -> str:
    """Translate Arabic queries to English for better retrieval accuracy."""
    if not text or not text.strip():
        return ""

    try:
        translated = MyMemoryTranslator(source="arabic", target="en-US").translate(
            text
        )

        if isinstance(translated, list):
            translated = " ".join(str(part) for part in translated)

        return translated if translated else text

    except Exception as e:
        print(f"Translation failed: {e}")
        return text


def has_arabic(text: str) -> bool:
    """Check if input string contains Arabic characters."""
    if not text:
        return False
    return bool(re.search(r"[\u0600-\u06FF]", text))