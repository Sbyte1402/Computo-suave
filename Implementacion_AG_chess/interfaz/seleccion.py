import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaOctal import CromosomaOctal

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaOctal], participantes: int=3)->List[CromosomaOctal]:
        pass

class SeleccionTorneo(Seleccion):
    def __init__(self, participantes=3)->None:
        super().__init__()
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: List[CromosomaOctal])-> List[CromosomaOctal]:
        ganadores = []
        for i in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda cromosoma: cromosoma.score)
            ganadores.append(ganador)
        return ganadores