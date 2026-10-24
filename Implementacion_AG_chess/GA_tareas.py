import copy
from Cromosomas import CromosomaDecimal
from interfaz.aptitud import AptitudCarga
from interfaz.seleccion import SeleccionRuleta
from interfaz.reproduccion import ReproduccionUnCruce
from interfaz.mutacion import MutacionDecimal

class GATareas:
    def __init__(self, num_inv: int, num_genes: int, hyperparams: dict):
        self.poblacion = [CromosomaDecimal(num_genes) for _ in range(num_inv)]
        self.num_inv = num_inv
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, variables: dict, max_generaciones: int = 500):
        for generacion in range(1, max_generaciones + 1):
            self.evaluar.evaluar_poblacion(self.poblacion, variables)
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            mejor_max_carga = self.poblacion[0].max_carga

            if generacion % 10 == 0:
                print(f"[Generación {generacion:04d}] M(x) Mínimo: {mejor_max_carga} (Aptitud: {self.poblacion[0].score:.5f})")

            if mejor_max_carga == 11:
                print(f"\n[TAREAS] ¡Solución óptima (M(x)={mejor_max_carga}) encontrada en la generación {generacion}!")
                break

            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            self.poblacion = copy.deepcopy(generacion_nueva)

if __name__ == "__main__":
    variables = {"tiempos": [8, 5, 7, 3, 6, 4]}
    num_genes = 6
    cant_ind = 50

    gaparams = {
        "Aptitud": AptitudCarga(),
        "Seleccion": SeleccionRuleta(),
        "Reproduccion": ReproduccionUnCruce(num_genes, CromosomaDecimal), # Pasamos la clase
        "Mutacion": MutacionDecimal(0.05),
        "por_remplazo": 0.5
    }

    algoritmo = GATareas(cant_ind, num_genes, gaparams)
    algoritmo.buscar_solucion(variables)
    
    print(f"\nTop 5 Soluciones (Problema Tareas):")
    for i, c in enumerate(algoritmo.poblacion[:5]):
        print(f"[{i+1:02d}] {c.genes} : M(x) = {c.max_carga} | Score = {c.score:.5f}")