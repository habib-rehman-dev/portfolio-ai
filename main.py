# main.py
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI  # <-- must be AsyncOpenAI, not OpenAI
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000" , "https://habib-dev-605.vercel.app/"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

SYSTEM_PROMPT = """You are an AI assistant answering questions about [Your Name]...
[... your bio ...]
"""


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


@app.post("/chat")
async def chat(payload: ChatRequest):
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[m.model_dump() for m in payload.messages],
    ]

    async def event_generator():
        try:
            stream = await client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b:free",
                messages=full_messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            print(f"OpenRouter stream failed: {e}")
            yield "\n\n[Something went wrong. Please try again.]"

    return StreamingResponse(event_generator(), media_type="text/plain")
