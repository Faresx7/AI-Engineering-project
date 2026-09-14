from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx
from src.core import http_client as http_client_module
import uvicorn
from src.routes import instagram, messenger, whatsapp


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client_module.http_client = httpx.AsyncClient(timeout=10)
    yield
    await http_client_module.http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.include_router(instagram.router)
app.include_router(messenger.router)
app.include_router(whatsapp.router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="127.0.0.1", port=8000)