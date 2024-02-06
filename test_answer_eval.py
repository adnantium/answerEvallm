import unittest
from unittest.mock import Mock, patch
import pytest
from answer_eval import (get_answer_eval_chain, ANSWER_EVAL_PROMPT_TEMPLATE, Evaluation,
                         PydanticOutputParser, build_answer_eval_chain)


class TestAnswerEval(unittest.TestCase):

    @patch('langchain.output_parsers.PydanticOutputParser')
    def test_build_answer_eval_chain_prompt(self, mock_pydantic_parser):
        model_name = "gpt-4"
        template = ANSWER_EVAL_PROMPT_TEMPLATE
        criterion = {
            "Completeness": "The extent to ...",
            "Correctness": "The accuracy of ...",
            "Grammar": "Proper use of ...",
        }

        # Mocking the dependencies
        mock_output_parser = Mock()
        mock_output_parser.get_format_instructions.return_value = "format_instructions"

        # Calling the function
        chain = build_answer_eval_chain(model_name=model_name,
                                        template=template,
                                        criterion=criterion,
                                        output_parser=mock_output_parser)

        # Assertions
        prompt_step = chain.steps[0]
        self.assertEqual(prompt_step.template, template)
        self.assertEqual(sorted(prompt_step.input_variables), sorted(["question", "answer"]))
        self.assertEqual(prompt_step.partial_variables["format_instructions"],
                         "format_instructions")
        for crit in criterion.keys():
            self.assertTrue(crit in prompt_step.partial_variables["criteria_list_text"])
        self.assertEqual(prompt_step.partial_variables["evaluator_name"], model_name)

        mock_output_parser.get_format_instructions.assert_called_once()
        mock_output_parser.reset_mock()

        # Calling the function again to test different inputs
        with pytest.raises(ValueError):
            chain = build_answer_eval_chain(model_name="gpt-DOESZNOTEXIST",
                                            template="Another template",
                                            criterion={},
                                            output_parser=mock_output_parser)

        with pytest.raises(AssertionError):
            chain = build_answer_eval_chain(model_name="gpt-4",
                                            template="Another template",
                                            criterion='INVALID',
                                            output_parser=mock_output_parser)

        mock_output_parser.reset_mock()


if __name__ == '__main__':
    unittest.main()
