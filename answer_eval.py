
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

DEFAULT_EVAL_CRITERIA = {
    "Completeness": "Does the answer cover all the important informaton expected in an excellent answer? What information is missing?",
    "Correctness": "Is answer free of errors and inaccurate information?",
    "Grammar": "Does it use proper english grammar and spelling?",
}

ANSWER_EVAL_PROMPT_TEMPLATE = """
    You are a college professor who is evaluating student's answers to test questions. 
    Rate the quality (on a scale of 0.0 to 1.0) of the answer based on the following evaluation criteria:

    1. Completeness: Does the answer cover all the important informaton expected in an excellent answer? What information is missing?
    2. Correctness: Is answer free of errors and inaccurate information?
    3. Grammar: Does it use proper english grammar and spelling?
    
    {criteria}

    Your response should only be in formatted JSON list of dictionaries with the attributes: 
    1. criteria
    2. score
    3. comments

    The "comments" attribute should only include critcisms and detailed suggestions for improvements.

    **Question**: "{question}"

    **Answer**: "{answer}"
    """


def get_answer_eval_prompt(template=ANSWER_EVAL_PROMPT_TEMPLATE):
    prompt = ChatPromptTemplate.from_messages([
        ("user", template)
    ])
    return prompt


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
    # output_parser = StrOutputParser()
    prompt = get_answer_eval_prompt()
    # criteria_list_text = get_criteria_list_text()
    model = ChatOpenAI()
    chain = prompt | model
    return chain


answer_eval_chain = get_answer_eval_chain()
inputs = {
    'question': 'What are the differences and similarities of plant and animal cells?',
    'answer': """Both plant and animal cells have many organelles in common. 
        They both include organelles such as ribosomes, mitochondrion, nuclei, and cell membranes. 
        Plant cells also contain a cell wall, which animal cells do not contain.""",
    # 'criteria': 'testing',
}
response = answer_eval_chain.invoke(inputs)
print(response)
pass
