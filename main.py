from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import httpx

from src.core import http_client as http_client_module
from src.routes import instagram, messenger, whatsapp
from src.rag.rag_chain import RAGChain
from src.core import container

@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client_module.http_client = httpx.AsyncClient(timeout=10)
    container.rag_chain_instance = RAGChain()
    yield
    await http_client_module.http_client.aclose()
    app.state.rag_chain = None

app = FastAPI(lifespan=lifespan)

app.include_router(instagram.router)
app.include_router(messenger.router)
app.include_router(whatsapp.router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="127.0.0.1", port=8000)