# AnswerEvaLLM

Tools for evaluating the quality of answer to a question.

```
curl --location 'http://localhost:8000/answer_eval/invoke/' \
--header 'Content-Type: application/json' \
--data '{
        "input": {
            "question": "What are the differences and similarities of plant and animal cells?",
            "answer": "Both plant and animal cells have many organelles in common. They both include organelles such as ribosomes, mitochondrion, nuclei, and cell membranes. Plant cells also contain a cell wall, which animal cells do not contain."
        }
    }'
```

![qa_form_eg.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8afb52d7-6b79-4e35-8269-5b5dec93eb43/96efa2ac-634f-40d1-ba49-3b5fae65c78d/qa_form_eg.png)
