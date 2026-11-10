import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CLASES DE LA RED NEURONAL DESDE CERO
# ==========================================
class Layer():
    def __init__(self, entradas: int, neuronas: int, activacion: str, bias: bool):
        self.Neuronas = neuronas
        self.Bias = bias
        self.funcion_activacion = activacion.lower()
        filas = entradas + 1 if bias else entradas
        # Inicialización de pesos
        self.Pesos = np.random.randn(filas, neuronas) * 0.1
        self._configurar_activacion(activacion)
    
    def _configurar_activacion(self, nombre: str):
        if nombre == "sigmoid":
            self.Activacion = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif nombre == "relu":
            self.Activacion = lambda x: np.maximum(0, x)
        elif nombre == "linear":
            self.Activacion = lambda x: x
        else:
            self.Activacion = lambda x: x

    def derivada_activacion(self) -> np.ndarray:
        if self.funcion_activacion == "sigmoid":
            return self.a_raw * (1 - self.a_raw)
        elif self.funcion_activacion == "relu":
            return np.where(self.z > 0, 1.0, 0.0)
        return np.ones_like(self.a_raw)

    def FeedForward(self, x: np.ndarray) -> np.ndarray:
        if self.Bias:
            self.entrada = np.hstack((x, np.ones((x.shape[0], 1))))
        else:
            self.entrada = x
        self.z = np.dot(self.entrada, self.Pesos)
        self.a_raw = self.Activacion(self.z)
        self.a = self.a_raw
        return self.a
    
    def BackPropagation(self, lr: float, is_last_layer: bool, next_w: np.ndarray, delta_siguiente: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
        m = self.entrada.shape[0]
        if is_last_layer:
            error = self.a - y_true
            self.delta = error * self.derivada_activacion()
        else:
            w_sin_bias = next_w[:-1, :] if (self.Bias and next_w.shape[0] > self.Neuronas) else next_w
            self.delta = np.dot(delta_siguiente, w_sin_bias.T) * self.derivada_activacion()
        
        self.gradiente_w = np.dot(self.entrada.T, self.delta) / m
        self.Pesos -= lr * self.gradiente_w
        return self.delta

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.FeedForward(output)
        return output
        
    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 50, lr: float = 0.01):
        if len(y.shape) == 1: y = y.reshape(-1, 1)
        for epoch in range(max_epochs):
            out = X
            for layer in self.layers: 
                out = layer.FeedForward(out)
            
            delta = None
            for i in reversed(range(len(self.layers))):
                is_last = (i == len(self.layers) - 1)
                next_w = self.layers[i + 1].Pesos if not is_last else None
                delta = self.layers[i].BackPropagation(lr, is_last, next_w, delta, y)
                
        # Calcular pérdida final (MSE)
        out = X
        for layer in self.layers: 
            out = layer.FeedForward(out)
        return np.mean((out - y)**2)

# ==========================================
# 2. ALGORITMO GENÉTICO PARA HIPERPARÁMETROS
# ==========================================
class CromosomaNN:
    def __init__(self):
        self.score = -1.0
        self.loss = float('inf')
        # Genes: [Num Capas Ocultas (1-2), Neuronas L1, Neuronas L2, Learning Rate, Activación]
        self.genes = [
            random.randint(1, 2),
            random.choice([8, 16, 32]),
            random.choice([8, 16, 32]),
            random.choice([0.1, 0.01, 0.05]),
            random.choice(["relu", "sigmoid"])
        ]

class AptitudNN:
    def evaluar_poblacion(self, poblacion, X_train, y_train, X_test, y_test):
        for crom in poblacion:
            num_layers, n1, n2, lr, act = crom.genes
            nn = NeuralNetwork()
            
            # Construir arquitectura basada en genes
            nn.add_layer(Layer(X_train.shape[1], n1, act, True))
            if num_layers == 2:
                nn.add_layer(Layer(n1, n2, act, True))
                nn.add_layer(Layer(n2, 1, "sigmoid", True))
            else:
                nn.add_layer(Layer(n1, 1, "sigmoid", True))
            
            # Entrenar
            loss = nn.fit(X_train, y_train, max_epochs=50, lr=lr)
            crom.loss = loss
            
            # Evaluar Accuracy
            preds = nn.predict(X_test)
            preds_bin = (preds > 0.5).astype(int)
            acc = np.mean(preds_bin.ravel() == y_test.ravel())
            
            # Fitness = Accuracy (buscamos maximizarlo)
            crom.score = acc 

def cruzar(potencial, pop_size):
    poblacion = []
    while len(poblacion) < pop_size:
        p1, p2 = random.sample(potencial, 2)
        h1, h2 = CromosomaNN(), CromosomaNN()
        punto = random.randint(1, 4)
        h1.genes = p1.genes[:punto] + p2.genes[punto:]
        h2.genes = p2.genes[:punto] + p1.genes[punto:]
        poblacion.extend([h1, h2])
    return poblacion[:pop_size]

def mutar(poblacion, prob=0.2):
    for c in poblacion:
        if random.random() < prob: c.genes[0] = random.randint(1, 2)
        if random.random() < prob: c.genes[1] = random.choice([8, 16, 32])
        if random.random() < prob: c.genes[2] = random.choice([8, 16, 32])
        if random.random() < prob: c.genes[3] = random.choice([0.1, 0.01, 0.05])
        if random.random() < prob: c.genes[4] = random.choice(["relu", "sigmoid"])

# ==========================================
# 3. EJECUCIÓN Y COMPARACIÓN POR DATASET
# ==========================================
datasets = {
    "Wine Quality": load_wine(return_X_y=True, as_frame=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True, as_frame=True)
}

for name, (X_df, y_s) in datasets.items():
    print(f"\n{'='*60}")
    print(f" DATASET: {name}")
    print(f"{'='*60}")
    
    # 1. Análisis Exploratorio Básico (EDA)
    print("1. ANÁLISIS EXPLORATORIO:")
    print(f" - Muestras: {X_df.shape[0]}, Variables: {X_df.shape[1]}")
    print(f" - Variables principales: {', '.join(X_df.columns[:5].tolist())}...")
    print(f" - Distribución de clases:\n{y_s.value_counts().to_string()}")
    
    # Binarizar objetivo
    y = (y_s > np.median(y_s)).astype(int).values
    X = X_df.values
    
    # Preprocesamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # 2. Arquitectura Base
    print("\n2. ARQUITECTURA BASE (1 Capa, 16 neuronas, LR=0.01, Sigmoid):")
    nn_base = NeuralNetwork()
    nn_base.add_layer(Layer(X_train.shape[1], 16, "sigmoid", True))
    nn_base.add_layer(Layer(16, 1, "sigmoid", True))
    
    loss_base = nn_base.fit(X_train, y_train, max_epochs=50, lr=0.01)
    preds_base = nn_base.predict(X_test)
    acc_base = np.mean((preds_base > 0.5).astype(int).ravel() == y_test.ravel())
    
    print(f" - Pérdida Total (MSE): {loss_base:.4f}")
    print(f" - Rendimiento (Accuracy): {acc_base:.4f}")

    # 3. Algoritmo Genético
    print("\n3. OPTIMIZACIÓN CON ALGORITMO GENÉTICO:")
    pop_size = 8
    generaciones = 5
    poblacion = [CromosomaNN() for _ in range(pop_size)]
    aptitud = AptitudNN()
    
    mejor_historico = None
    
    for gen in range(generaciones):
        aptitud.evaluar_poblacion(poblacion, X_train, y_train, X_test, y_test)
        poblacion.sort(key=lambda x: x.score, reverse=True)
        
        if mejor_historico is None or poblacion[0].score > mejor_historico.score:
            mejor_historico = poblacion[0]
            
        print(f" - Gen {gen+1:02d} | Mejor Accuracy: {poblacion[0].score:.4f} | Pérdida: {poblacion[0].loss:.4f}")
        
        # Selección Torneo (de a 2)
        potencial = [max(random.sample(poblacion, 2), key=lambda x: x.score) for _ in range(pop_size)]
        poblacion = cruzar(potencial, pop_size)
        mutar(poblacion, prob=0.2)
        
    print("\n4. COMPARACIÓN FINAL:")
    print(" BASE:")
    print(f"  -> Accuracy: {acc_base:.4f} | Loss: {loss_base:.4f}")
    print(" OPTIMIZADA (GENÉTICA):")
    print(f"  -> Accuracy: {mejor_historico.score:.4f} | Loss: {mejor_historico.loss:.4f}")
    print("  -> Hiperparámetros encontrados:")
    print(f"     Capas Ocultas: {mejor_historico.genes[0]}")
    print(f"     Neuronas L1: {mejor_historico.genes[1]} | Neuronas L2: {mejor_historico.genes[2]}")
    print(f"     Learning Rate: {mejor_historico.genes[3]}")
    print(f"     Activación: {mejor_historico.genes[4]}")