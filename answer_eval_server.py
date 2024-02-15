#!/usr/bin/env python
from typing import List
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from fastapi.responses import HTMLResponse, JSONResponse

from answer_eval import Evaluation, CriteriaScore, get_answer_eval_chain

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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/qa", response_class=HTMLResponse)
async def qa_form(request: Request):
    return templates.TemplateResponse("qa_form.html", {"request": request})


@app.post("/qa", response_class=HTMLResponse)
async def qa_form_evaluate(request: Request, question: str = Form(...), answer: str = Form(...)):
    inputs = {
        'question': question,
        'answer': answer,
    }
    evaluation: Evaluation = get_answer_eval_chain().invoke(inputs)
    response_data = {"request": request, **inputs, "evaluation": evaluation}
    return templates.TemplateResponse("qa_form.html", response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
