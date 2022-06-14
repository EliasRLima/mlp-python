from fastapi import FastAPI
from Model import AmostraEntenty
from Controller import TreinadorController
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def root():
    return {"PERCEPTRON": "Bem vindo ao algortimo perceptron"}


@app.post('/separar')
def upload_file_and_read(dataset: AmostraEntenty.Dataset):
    return {"PERCEPTRON": TreinadorController.separar(dataset.conteudo)}