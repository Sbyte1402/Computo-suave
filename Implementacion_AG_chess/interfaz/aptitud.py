from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[Any], var: Dict = None):
        pass

class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[Any], var: Dict = None):
        for cromosoma in poblacion:
            conflictos = 0
            for i in range(cromosoma.n_genes):
                for j in range(i + 1, cromosoma.n_genes):
                    if cromosoma.genes[i] == cromosoma.genes[j]:
                        conflictos += 1
                    elif abs(cromosoma.genes[i] - cromosoma.genes[j]) == abs(i - j):
                        conflictos += 1
            cromosoma.score = 28 - conflictos

class AptitudCarga(Aptitud):
    def evaluar_poblacion(self, poblacion: List[Any], var: Dict = None):
        tiempos = var["tiempos"]
        for cromosoma in poblacion:
            cargas = {1: 0, 2: 0, 3: 0}
            for i in range(cromosoma.n_genes):
                procesador = cromosoma.genes[i]
                cargas[procesador] += tiempos[i]
                
            max_carga = max(cargas.values())
            cromosoma.max_carga = max_carga
            # f(x) = 1 / (1 + M(x))
            cromosoma.score = 1.0 / (1.0 + max_carga)