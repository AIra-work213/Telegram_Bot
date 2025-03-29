import model
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def func():
    pass