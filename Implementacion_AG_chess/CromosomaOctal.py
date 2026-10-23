import random

class CromosomaOctal(object):
    def __init__(self, genes: int = 8):
        self.n_genes: int = genes
        self.score: float = -1.0
        self.genes: list = [0] * genes

    def llenar_random_data(self):
        """Asigna un valor aleatorio (0-7) a cada gen (columna)"""
        for i in range(self.n_genes):
            # Para el problema de 8 reinas, los valores son de 0 a 7
            self.genes[i] = random.randint(0, 7)

    def get_gen(self, gen_index: int) -> int:
        return self.genes[gen_index]

    def set_gen(self, valor: int, gen_index: int) -> None:
        self.genes[gen_index] = valor