from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from app.llm import generate_text

app = FastAPI()

@app.post("/api/chat")
def chat(prompt: str = Form(...)):
    response = generate_text(prompt)
    return JSONResponse({"response": response})
