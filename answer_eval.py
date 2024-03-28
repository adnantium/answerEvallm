from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables.base import Runnable
from typing import Dict, List
import langchain

langchain.debug = True

ANSWER_EVAL_PROMPT_TEMPLATE = """
You are a college professor who is evaluating student's answers to test questions. 

Rate the quality of the answer with a score between 0 and 100 based on each of the following evaluation criteria based on thier definitions:

{criteria_list_text}

Use the following scale to score the answer for each criteria:
* 100: Excellent. The answer fully meets the criteria's definition.
* 90-99: Very Good. The answer is very good but not perfect.
* 80-89: Good. The answer has some minor issues.
* 70-79: Fair. The answer has some major issues.
* 60-69: Poor. The answer has many issues.
* 0-59: Very Poor. The answer does not meet the criteria's definition.

The "comments" should be concise sentence fragments of constructive criticisms and detailed, specific recommendations for enhancements or correction. Provide the 
correct answer if the answer is incorrect or incomplete. Justify the score given 
for each criteria if it not 100.
        
{format_instructions}
    
The "evaluator_name" field value should be "{evaluator_name}"

**Question**: "{question}"

**Answer**: "{answer}"
"""

DEFAULT_EVAL_CRITERION = {
    "Completeness": "The extent to which the answer covers all aspects of the question",
    "Correctness": "The accuracy of the information provided in the answer",
    "Grammar": "Proper use of grammar, punctuation, correct spellings and must be full sentences",
}


class CriteriaScore(BaseModel):
    """
    Represents the evaluation criteria, score, and comments for an answer.
    """
    name: str = Field(description="The name of the criteria e.g Grammar")
    definition: str = Field(description="The definition used for the criteria")
    comments: str = Field(description="Helpful comments and potential improvements")
    score: int = Field(description="The score for the given criteria")


class Evaluation(BaseModel):
    """
    Represents the evaluation of an answer across all criteria.
    """
    evaluator_name: str = Field(description="Name of the system that did the evaluation")
    scores: List[CriteriaScore] = Field(description="List of scores for each criteria")


def build_answer_eval_chain(model_name: str, template: str, criterion: Dict[str, int],
                            output_parser: BaseOutputParser) -> Runnable:
    """
    Contructs a runnable chain for evaluating an answer.
    """
    model: ChatOpenAI = get_eval_model(model_name)
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "answer"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions(),
            "criteria_list_text": build_criterion_list_text(criterion),
            "evaluator_name": model.model_name,
        },
    )
    chain = prompt | model | output_parser
    return chain


def get_answer_eval_chain() -> Runnable:
    """
    Specifies the model, template, criterion, and output parser 
    used to construct a runnable chain.
    """
    model_name: str = 'gpt-4'
    template: str = ANSWER_EVAL_PROMPT_TEMPLATE
    criterion: dict = DEFAULT_EVAL_CRITERION
    pydantic_parser = PydanticOutputParser(pydantic_object=Evaluation)

    chain = build_answer_eval_chain(model_name, template, criterion, pydantic_parser)
    return chain


def get_eval_model(model_name: str, temperature: int = 0) -> ChatOpenAI:
    """
    Gets an instance of ChatOpenAI class for the model with a temperature of 0.
    """
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    return model


def build_criterion_list_text(criterion: Dict[str, str]) -> str:
    """
    Takes a dictionary with key: "Criteria name" and value: "Criteria Definition".
    It builds a formatted string as a numeric list that can be used in a prompt template.

    Example:    
    >>> build_criterion_list_text({'Completeness': 'The extent to ...', 'Grammar': \
        'The definition of ...'})
    '1. Completeness: The extent to which the answer covers all aspects of the question\n`
    '2. Grammar: The extent to ... \n'
    """
    assert type(criterion) is dict, "Criterion must be a dictionary"
    formated_text = ""
    for index, criteria in enumerate(criterion.items()):
        formated_text += f"  {index+1}. {criteria[0]}: {criteria[1]}\n"
        # e.g. '1. Completeness: The extent to which the answer ...\n`
    return formated_text
