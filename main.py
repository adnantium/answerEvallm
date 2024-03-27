from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# # Serve static files like CSS and JavaScript
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup template directory
templates = Jinja2Templates(directory="templates")


@app.get("/qa", response_class=HTMLResponse)
async def get_form(request: Request):
    # Render the HTML form
    return templates.TemplateResponse("qa_form.html", {"request": request})


@app.post("/qa/evaluate")
async def evaluate(question: str = Form(...), answer: str = Form(...)):
    # Process the form data here. For example, return a simple JSON
    return JSONResponse(content={
        "question": question,
        "answer": answer,
        "evaluation": "This is a sample response."
    })
