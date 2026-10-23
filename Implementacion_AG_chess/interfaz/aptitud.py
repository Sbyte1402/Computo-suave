from abc import ABC, abstractmethod
from typing import List, Dict
from CromosomaOctal import CromosomaOctal

class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict = None):
        """Interfaz encargada para darle un score a la población"""
        pass

class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict = None):
        for cromosoma in poblacion:
            conflictos = 0
            # Comparar todos los pares de reinas
            for i in range(cromosoma.n_genes):
                for j in range(i + 1, cromosoma.n_genes):
                    # Conflicto: misma fila
                    if cromosoma.genes[i] == cromosoma.genes[j]:
                        conflictos += 1
                    # Conflicto: misma diagonal
                    elif abs(cromosoma.genes[i] - cromosoma.genes[j]) == abs(i - j):
                        conflictos += 1
            
            # Función Fitness: f(x) = 28 - C(x)
            cromosoma.score = 28 - conflictos