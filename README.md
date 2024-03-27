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

![qa_form_eg.png](docs/qa_form_eg.png)
