
from rich import print
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
import json

import langchain
langchain.debug = True

# 1. Completeness: Does the answer cover all the important informaton expected in an excellent answer? What information is missing?
# 2. Correctness: Is answer free of errors and inaccurate information?
# 3. Grammar: Does it use proper english grammar and spelling?


DEFAULT_EVAL_CRITERIA = ["Completeness", "Correctness", "Grammar",]
# DEFAULT_EVAL_CRITERIA = {
#     "Completeness": "Does the answer cover all the important informaton expected in an excellent answer? What information is missing?",
#     "Correctness": "Is answer free of errors and inaccurate information?",
#     "Grammar": "Does it use proper english grammar and spellings?",
# }

ANSWER_EVAL_PROMPT_TEMPLATE = """
    You are a college professor who is evaluating student's answers to test questions. 
    
    Rate the quality with a score between 0 and 100 of the answer based on each of the following evaluation criteria and its definition:

    {criteria_list_text}

    The "comments" should be concise, addressed directly to the author, and should exclusively comprise constructive criticisms and detailed, specific recommendations for enhancements.
    
    {format_instructions}

    **Question**: "{question}"

    **Answer**: "{answer}"
    """


class CriteriaScore(BaseModel):
    name: str = Field(description="The name of the criteria e.g Grammar")
    definition: str = Field(description="The definition used for the criteria")
    # evaluator: str = Field(description="Name of the system that did the evaluation")
    comments: str = Field(
        description="Helpful comments on any problems with the answer and potentional improvments")
    score: int = Field(
        description="The scrore given to the answer for the given criteria")


class Evaluation(BaseModel):
    question: str = Field(description="The question that was asked")
    answer: str = Field(description="The answer given for the question")
    scores: List[CriteriaScore] = Field(
        description="List of scores for each evaluation criteria")

def get_criteria_list_text(criterion=None):
    """Takes dict (with specific requirements) or list/tuple

    Args:
        criterion (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    criterion = criterion or DEFAULT_EVAL_CRITERIA
    criteria_list = ""
    if type(criterion) is dict:
        for i, c in enumerate(criterion.items()):
            criteria_list += f"{i+1}. {c[0]}: {c[1]}\n"
    else:
        for i, c in enumerate(criterion):
            criteria_list += f"{i+1}. {c}\n"

    return criteria_list


def get_answer_eval_chain():
    template = ANSWER_EVAL_PROMPT_TEMPLATE
    pydantic_parser = PydanticOutputParser(pydantic_object=Evaluation)
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "answer"],
        partial_variables={
            "format_instructions": pydantic_parser.get_format_instructions(),
            "criteria_list_text": get_criteria_list_text()
        },
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | model | pydantic_parser
    return chain


# answer_eval_chain = get_answer_eval_chain()
# inputs = {
#     'question': 'What are the differences and similarities of plant and animal cells?',
#     'answer': """Both plant and animal cells have many organelles in common.
#         They both include organelles such as ribosomes, mitochondrion, nuclei, and cell membranes.
#         Plant cells also contain a cell wall, which animal cells do not contain.""",
#     'criteria': '',
# }
# response: Evaluation = answer_eval_chain.invoke(inputs)
# # response_js = response.json()
# # print(json.dumps(response_js, indent=2))
# print(response)
# pass
