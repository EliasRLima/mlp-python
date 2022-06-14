
from pydantic import BaseModel
from typing import Set


class Dataset(BaseModel):
    titulo: str
    conteudo: Set[str] = []


class Amostra(BaseModel):
    nome: str
    x1: float
    x2: float
    x3: float
    d: int