import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class AIModel:
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
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.sys_prompt,
                temperature=0.6,
                top_p=0.9,
            )
        )

    def generate_response(self, prompt: str) -> str:
        # Send message using the managed chat session
        response = self.chat.send_message(prompt)
        return response.text or ""

    def generate_stream_response(self, prompt: str):
        # Stream response chunks in real-time
        response_stream = self.chat.send_message_stream(prompt)
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text