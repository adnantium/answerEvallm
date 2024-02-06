from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables.base import RunnableSerializable, Runnable
from typing import Dict, List, Union
from rich import print
import json
import langchain

langchain.debug = True

ANSWER_EVAL_PROMPT_TEMPLATE = """
You are a college professor who is evaluating student's answers to test questions. 

Rate the quality of the answer with a score between 0 and 100 based on each of the following evaluation criteria and its definition:

{criteria_list_text}

The "comments" should be concise sentence fragments of constructive criticisms and detailed, specific recommendations for enhancements or correction. Provide the correct answer if the answer is incorrect or incomplete. Justify the score given for each criteria.
        
{format_instructions}
    
The "evaluator_name" field value should be "{evaluator_name}"

**Question**: "{question}"

**Answer**: "{answer}"
"""

DEFAULT_EVAL_CRITERION = {
    "Completeness": "The extent to which the answer covers all aspects of the question",
    "Correctness": "The accuracy of the information provided in the answer",
    "Grammar": "Proper use of grammar, punctuation and correct spellings",
}


class CriteriaScore(BaseModel):
    """
    Represents the evaluation criteria, score, and comments for an answer.
    """
    name: str = Field(description="The name of the criteria e.g Grammar")
    definition: str = Field(description="The definition used for the criteria")
    comments: str = Field(
        description="Helpful comments on any problems with the answer and potential improvements")
    score: int = Field(description="The score given to the answer for the given criteria")


class Evaluation(BaseModel):
    evaluator_name: str = Field(description="Name of the system that did the evaluation")
    scores: List[CriteriaScore] = Field(description="List of scores for each evaluation criteria")


class QAEvaluation(BaseModel):
    question: str = Field(description="The question that was asked")
    answer: str = Field(description="The answer given for the question")
    evaluations: List[Evaluation] = Field(description="Evaluation scores from multiple systems")


def build_criterion_list_text(criterion: Dict[str, str]) -> str:
    """
    Takes a criterion as dictionary with string keys and string values.
    It builds a formatted string of the criterion in a bullet list format that can 
    be used in a prompt template.

    If no criterion is provided, the function uses a default criterion (DEFAULT_EVAL_CRITERION).

    Parameters:
    criterion (Dict[str, str]]): A criterion represented as a list or a dictionary. 
    Defaults to None.

    Returns:
    str: A formatted numbered list string that represents the criterion.

    Example:    
    >>> build_criterion_list_text({'Completeness': 'The extent to ...', 'Grammar': 'The extent to ...'})
    '1. Completeness: The extent to which the answer covers all aspects of the question\n`
    '2. Grammar: The extent to ... \n'
    """
    assert type(criterion) is dict

    formated_text = ""
    for index, criteria in enumerate(criterion.items()):
        formated_text += f"  {index+1}. {criteria[0]}: {criteria[1]}\n"
        # e.g. '1. Completeness: The extent to which the answer ...\n`

    return formated_text


AVAILABLE_MODELS = [
    'gpt-4',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-1106',
    'gpt-4-turbo-preview',
]


def get_eval_model(model_name: str) -> ChatOpenAI:
    """
    Gets an instance of an evaluation model. Creates default model if no model 
    name is provided.

    Args:
        model_name (str, optional): The name of the model to be instantiated. 
        Defaults to None.

    Raises:
        ValueError: If the provided model name is not in the list of 
        available models.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class with the specified 
        model name and a temperature of 0.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model name must be one of {AVAILABLE_MODELS}")
    model = ChatOpenAI(model_name=model_name, temperature=0)
    return model


def build_answer_eval_chain(model_name: str, template: str, criterion: Dict[str, int],
                            output_parser: BaseOutputParser) -> Runnable:
    model: ChatOpenAI = get_eval_model(model_name)
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "answer"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions(),
            "criteria_list_text": build_criterion_list_text(criterion),
            "evaluator_name": model.model_name
        },
    )
    chain = prompt | model | output_parser
    return chain


def get_answer_eval_chain() -> Runnable:
    model_name: str = 'gpt-4'
    template: str = ANSWER_EVAL_PROMPT_TEMPLATE
    criterion = DEFAULT_EVAL_CRITERION
    pydantic_parser = PydanticOutputParser(pydantic_object=Evaluation)

    chain = build_answer_eval_chain(model_name, template, criterion, pydantic_parser)
    return chain


def evaluate_answer_multi_models(question: str, answer: str, model_names: List[str] = None):
    model_names = model_names or AVAILABLE_MODELS
    chains = [get_answer_eval_chain(model_name) for model_name in model_names]
    inputs = {
        'question': question,
        'answer': answer,
    }
    all_evaluations = [chain.invoke(inputs) for chain in chains]
    return all_evaluations


if __name__ == '__main__':

    question = 'What are the differences and similarities of plant and animal cells?'
    answer = """Both plant and animal cells have many organelles in common.
        They both include organelles such as ribosomes, mitochondrion, nuclei, and cell membranes.
        Plant cells also contain a cell wall, which animal cells do not contain."""

    # answer = """Plant and animal cells, both eukaryotic, share similarities like a nucleus, mitochondria, and endoplasmic
    # reticulum. However, they differ significantly: plant cells have a rigid cell wall, chloroplasts for photosynthesis,
    # and large central vacuoles for storage and structural support. Animal cells lack these structures but have centrioles
    # involved in cell division and smaller vacuoles. Additionally, plant cells often have a fixed rectangular shape due to
    # the cell wall, while animal cells have a more flexible and varied shape. Both types of cells play crucial roles in their
    # respective organisms' survival and functioning."""

    answer_eval_chain = get_answer_eval_chain()
    inputs = {
        'question': question,
        'answer': answer,
    }
    response: Evaluation = answer_eval_chain.invoke(inputs)
    print(response)

    # all_evals = evaluate_answer_multi_models(question, answer)
    # print(all_evals)
    pass
