import random
import copy
from typing import List, Dict
from CromosomaOctal import CromosomaOctal
from interfaz.aptitud import AptitudReinas
from interfaz.seleccion import SeleccionTorneo
from interfaz.reproduccion import ReproduccionUnCruce
from interfaz.mutacion import MutacionOctal

class GA(object):
    def __init__(self, num_inv:int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaOctal] = [CromosomaOctal(num_genes) for _ in range(num_inv)]
        self.num_inv:int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"]*num_inv)
        
        # Inicializar población aleatoria
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, max_generaciones: int = 1000):
        generacion: int = 1
        solucion_encontrada = False

        while generacion <= max_generaciones:
            # 1. Evaluar
            self.evaluar.evaluar_poblacion(self.poblacion)
            
            # 2. Ordenar por score
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            mejor_score = self.poblacion[0].score

            # Mostrar progreso
            if generacion % 10 == 0:
                print(f"[Generación {generacion:04d}] Mejor Score: {mejor_score}/28")

            # 3. Condición de parada: Un score de 28 significa 0 conflictos (solución perfecta)
            if mejor_score == 28:
                print(f"\n¡Solución óptima encontrada en la generación {generacion}!")
                solucion_encontrada = True
                break

            # 4. Selección, Reproducción y Mutación
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)
            
        if not solucion_encontrada:
            print("\nLímite de generaciones alcanzado sin encontrar el óptimo global.")

if __name__ == "__main__":
    num_genes = 8       # 8 Columnas
    cant_ind = 150      # Tamaño de la población (Aumentado ligeramente para mayor diversidad)

    gaparams = {}
    gaparams["Aptitud"] = AptitudReinas()
    gaparams["Seleccion"] = SeleccionTorneo(3)  # Torneo con 3 participantes
    gaparams["Reproduccion"] = ReproduccionUnCruce(num_genes)
    gaparams["Mutacion"] = MutacionOctal(0.1)   # 10% probabilidad de mutación
    gaparams["por_remplazo"] = 0.5 # Conserva la mitad por elitismo

    algoritmoGA = GA(cant_ind, num_genes, gaparams)
    
    # Inicia la búsqueda
    algoritmoGA.buscar_solucion(max_generaciones=2000)
    
    # Imprimir los mejores resultados
    print(f"\n{'Top':<4} | {'Cromosoma (Fila por columna)':<35} | {'Score'}")
    print("-" * 55)
    for i, cromosoma in enumerate(algoritmoGA.poblacion[:10]):
        print(f"[{i+1:02d}]  {str(cromosoma.genes):<35} : {cromosoma.score}")