from abc import ABC, abstractmethod
from typing import List, Any
import random

class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[Any]):
        pass

class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[Any]):
        for crom in poblacion:
            for i in range(crom.n_genes):
                if random.random() < self.prob:
                    crom.set_gen(random.randint(0, 7), i)

class MutacionDecimal(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[Any]):
        for crom in poblacion:
            for i in range(crom.n_genes):
                if random.random() < self.prob:
                    crom.set_gen(random.randint(1, 3), i)