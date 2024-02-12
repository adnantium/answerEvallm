#!/usr/bin/env python
from typing import List
from langserve import add_routes
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from fastapi.responses import HTMLResponse, JSONResponse

from answer_eval import Evaluation, get_answer_eval_chain

import langchain

langchain.debug = True

app = FastAPI(
    title="AnswerEvallm",
    version="1.0",
    description="An API for evaluating the quality of the answer to a question",
)

add_routes(
    app,
    get_answer_eval_chain(),
    path="/answer_eval",
    output_type=Evaluation,
)

templates = Jinja2Templates(directory="templates")


@app.get("/qa", response_class=HTMLResponse)
async def get_form(request: Request):
    # Render the HTML form
    return templates.TemplateResponse("qa_form.html", {"request": request})


@app.post("/qa/evaluate")
async def evaluate(question: str = Form(...), answer: str = Form(...)):
    answer_eval_chain = get_answer_eval_chain()
    inputs = {
        'question': question,
        'answer': answer,
    }
    response: Evaluation = answer_eval_chain.invoke(inputs)
    print(response)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
