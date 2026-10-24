import random

class CromosomaOctal:
    def __init__(self, genes: int = 8):
        self.n_genes: int = genes
        self.score: float = -1.0
        self.genes: list = [0] * genes

    def llenar_random_data(self):
        for i in range(self.n_genes):
            self.genes[i] = random.randint(0, 7) # Fila 0 a 7

    def get_gen(self, gen_index: int) -> int:
        return self.genes[gen_index]

    def set_gen(self, valor: int, gen_index: int) -> None:
        self.genes[gen_index] = valor

class CromosomaDecimal:
    def __init__(self, genes: int = 6):
        self.n_genes: int = genes
        self.score: float = -1.0
        self.genes: list = [0] * genes
        self.max_carga: int = 0

    def llenar_random_data(self):
        for i in range(self.n_genes):
            self.genes[i] = random.randint(1, 3) # Procesador 1 a 3

    def get_gen(self, gen_index: int) -> int:
        return self.genes[gen_index]

    def set_gen(self, valor: int, gen_index: int) -> None:
        self.genes[gen_index] = valor