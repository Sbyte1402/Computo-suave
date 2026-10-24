import random
from abc import ABC, abstractmethod
from typing import List, Any

class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[Any], participantes: int=3)->List[Any]:
        pass

class SeleccionTorneo(Seleccion):
    def __init__(self, participantes=3):
        super().__init__()
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: List[Any])-> List[Any]:
        ganadores = []
        for i in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda c: c.score)
            ganadores.append(ganador)
        return ganadores

class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion: List[Any]) -> List[Any]:
        seleccionados = []
        suma_aptitud = sum(c.score for c in poblacion)
        
        for _ in range(len(poblacion)):
            r = random.uniform(0, suma_aptitud)
            acumulado = 0
            for cromosoma in poblacion:
                acumulado += cromosoma.score
                if acumulado >= r:
                    seleccionados.append(cromosoma)
                    break
        return seleccionados