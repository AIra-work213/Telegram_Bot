import model
from fastapi import FastAPI

app = FastAPI()

@app.post('/')
def func(x):
    y =  model.model(x.unsqueeze(0)).squeeze(0)
    categories = model.description
    response = [[categories[i], 0] for i in range(len(categories))]
    for i in range(len(y)):
        response[i][1]=y[i]
    response = sorted(response, key=lambda x: x[1], reverse=True)
    return response
