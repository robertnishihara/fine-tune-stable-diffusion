"""Simple stable diffusion app.

How to run:
    $ uvicorn server:app --reload

How to test:
"""


from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}