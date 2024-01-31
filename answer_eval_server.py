#!/usr/bin/env python
from typing import List
from langserve import add_routes
from fastapi import FastAPI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence

from answer_eval import get_answer_single_eval_chain, Evaluation

import langchain

langchain.debug = True

app = FastAPI(
    title="AnswerEvallm",
    version="1.0",
    description="An API for evaluating the quality of the answer to a question",
)

add_routes(
    app,
    get_answer_single_eval_chain(),
    path="/answer_eval",
    output_type=Evaluation,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
