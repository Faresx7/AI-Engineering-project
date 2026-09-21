from dotenv import load_dotenv
from google.genai import types
from google import genai
import os

load_dotenv()

class AIModel:
    # ! sys prompt need to handled in a better way
    def __init__(self, sys_prompt='''Your name is Mr.Roberto.
                                    You are a sharp, smart, and friendly human.
                                    You pay close attention to details, stay calm,
                                    and talk naturally without overreacting.
                                    you may be asked for a products, if you have more than one product in input,
                                    tell the user all options that you have in the same range he is asking for'''):

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = "gemini-3.1-flash-lite"
        self.sys_prompt = sys_prompt
        
        # Initialize Gemini chat session with system instruction
        self.chat = self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.sys_prompt,
                temperature=0.6,
                top_p=0.9,
            )
        )

    async def generate_response(self, prompt: str) -> str:
        # Send message using the managed chat session
        response = await self.chat.send_message(prompt)
        return response.text or ""

    async def generate_stream_response(self, prompt: str):
        # Stream response chunks in real-time
        response_stream = await self.chat.send_message_stream(prompt)

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text