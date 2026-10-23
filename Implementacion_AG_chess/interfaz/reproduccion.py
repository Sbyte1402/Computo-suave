from abc import ABC, abstractmethod
from typing import List
from CromosomaOctal import CromosomaOctal
import random

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaOctal], potencial: List[CromosomaOctal], max_poblacion: int)->None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaOctal], potencial: List[CromosomaOctal], max_poblacion: int)->None:
        curr_po = len(poblacion)
        # Ordenar tipo elite, mayor a menor
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaOctal(self.num_genes)
            hijoB = CromosomaOctal(self.num_genes)
            
            # Punto de cruce (entre el índice 1 y 6 para 8 genes)
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % max_poblacion]
            padreB = ord_padres[(id_padres+1) % max_poblacion]

            # Cruzar listas
            for i in range(self.num_genes):
                if i < punto_cruce:
                    hijoA.set_gen(padreA.get_gen(i), i)
                    hijoB.set_gen(padreB.get_gen(i), i)
                else:
                    hijoA.set_gen(padreB.get_gen(i), i)
                    hijoB.set_gen(padreA.get_gen(i), i)

            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1

            id_padres += 1