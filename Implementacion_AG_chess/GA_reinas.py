import copy
from Cromosomas import CromosomaOctal
from interfaz.aptitud import AptitudReinas
from interfaz.seleccion import SeleccionTorneo
from interfaz.reproduccion import ReproduccionUnCruce
from interfaz.mutacion import MutacionOctal

class GAReinas:
    def __init__(self, num_inv: int, num_genes: int, hyperparams: dict):
        self.poblacion = [CromosomaOctal(num_genes) for _ in range(num_inv)]
        self.num_inv = num_inv
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, max_generaciones: int = 2000):
        for generacion in range(1, max_generaciones + 1):
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            mejor_score = self.poblacion[0].score

            if generacion % 10 == 0:
                print(f"[Generación {generacion:04d}] Mejor Score: {mejor_score}/28")

            if mejor_score == 28:
                print(f"\n[REINAS] ¡Solución óptima encontrada en la generación {generacion}!")
                break

            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            self.poblacion = copy.deepcopy(generacion_nueva)

if __name__ == "__main__":
    num_genes = 8
    cant_ind = 150

    gaparams = {
        "Aptitud": AptitudReinas(),
        "Seleccion": SeleccionTorneo(3),
        "Reproduccion": ReproduccionUnCruce(num_genes, CromosomaOctal), # Pasamos la clase
        "Mutacion": MutacionOctal(0.1),
        "por_remplazo": 0.5
    }

    algoritmo = GAReinas(cant_ind, num_genes, gaparams)
    algoritmo.buscar_solucion()
    
    print(f"\nTop 5 Soluciones (Problema Reinas):")
    for i, c in enumerate(algoritmo.poblacion[:5]):
        print(f"[{i+1:02d}] {c.genes} : {c.score}/28")