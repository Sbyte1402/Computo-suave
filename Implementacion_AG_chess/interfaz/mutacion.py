from abc import ABC, abstractmethod
from typing import List
from CromosomaOctal import CromosomaOctal
import random

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass

class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        for crom in poblacion:
            for i in range(crom.n_genes):
                prob_gen = random.random()
                if prob_gen < self.prob:
                    # Mutación octal: asigna una nueva fila aleatoria
                    nuevo_valor = random.randint(0, 7)
                    crom.set_gen(nuevo_valor, i)