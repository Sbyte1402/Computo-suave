from abc import ABC, abstractmethod
from typing import List, Any
import random

class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[Any], potencial: List[Any], max_poblacion: int)->None:
        pass

class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int, tipo_cromosoma: Any):
        super().__init__()
        self.num_genes = num_genes
        self.tipo_cromosoma = tipo_cromosoma # Inyectamos la clase a instanciar

    def reproduccion_poblacion(self, poblacion: List[Any], potencial: List[Any], max_poblacion: int)->None:
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = self.tipo_cromosoma(self.num_genes)
            hijoB = self.tipo_cromosoma(self.num_genes)
            
            punto_cruce = random.randint(1, self.num_genes - 1)

            padreA = ord_padres[id_padres % max_poblacion]
            padreB = ord_padres[(id_padres+1) % max_poblacion]

            for i in range(self.num_genes):
                if i < punto_cruce:
                    hijoA.set_gen(padreA.get_gen(i), i)
                    hijoB.set_gen(padreB.get_gen(i), i)
                else:
                    hijoA.set_gen(padreB.get_gen(i), i)
                    hijoB.set_gen(padreA.get_gen(i), i)

            if curr_po < max_poblacion: poblacion.append(hijoA); curr_po += 1
            if curr_po < max_poblacion: poblacion.append(hijoB); curr_po += 1

            id_padres += 1