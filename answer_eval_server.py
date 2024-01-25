#!/usr/bin/env python
from fastapi import FastAPI
# from langchain.chat_models import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence

from langserve import add_routes

from answer_eval import get_answer_eval_chain


app = FastAPI(
    title="AnswerEvallm",
    version="1.0",
    description="An API for evaluating the quality of the answer to a question",
)

add_routes(
    app,
    get_answer_eval_chain(),
    path="/answer_eval",
)

joke_model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
joke_chain = prompt | joke_model
add_routes(
    app,
    joke_chain,
    path="/joke",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
