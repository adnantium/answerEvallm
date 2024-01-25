
import json
from pprint import pprint

from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from answer_eval import AnswerEval


def evaluate_response(question: str, response: str, criterion: dict = None):

    llm = OpenAI()

    # check inputs

    # built prompt

    # build chain
    output_parser = StrOutputParser()
    prompt = AnswerEval.get_answer_eval_prompt()
    criteria_list_text = AnswerEval.get_criteria_list_text(criterion)

    # invoke chain
    chain_data = {
        "question": question,
        'answer': answer,
        'criteria_list': criteria_list_text
    }

    chain = prompt | llm | output_parser
    response = chain.invoke(chain_data)
    print(response)

    # parse message response
    response_data = json.loads(response)
    pprint(response_data)

    # format evaluation

    # return evaluation

    pass


question = """What are the differences and similarities of 
    plant and animal cells?"""
answer = """Both plant and animal cells have many organelles 
    in common. They both include organelles such as ribosomes, 
    mitochondrion, nuclei, and cell membranes. Plant cells also 
    contain a cell wall, which animal cells do not contain."""

response = evaluate_response(question, answer)
print(response)
