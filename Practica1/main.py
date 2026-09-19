import numpy as np

class Layer():
    def __init__(self, entradas: int, neuronas: int, activacion: str, bias: bool, optimizador: str):
        self.Neuronas = neuronas
        self.Bias = bias
        self.optimizador = optimizador
        
        filas = entradas + 1 if bias else entradas
        self.Pesos = np.ones((filas, neuronas)) * 0.1
        self._configurar_activacion(activacion)
    
    def _configurar_activacion(self, nombre: str):
        nombre = nombre.lower()
        if nombre == "sigmoid":
            self.Activacion = lambda x: 1 / (1 + np.exp(-x))
        elif nombre == "relu":
            self.Activacion = lambda x: np.maximum(0, x)
        elif nombre == "leaky_relu":
            self.Activacion = lambda x: np.where(x > 0, x, x * 0.01)
        elif nombre == "softmax":
            self.Activacion = lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(
                np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True
            )
        else:
            raise ValueError(f"Activación {nombre} no soportada.")

    def transfer_derivative(self, output: float) -> float:
        return np.where(output > 0, 1, 0)

    def FeedForward(self, x: np.ndarray) -> np.ndarray:
        if self.Bias:
            ones = np.ones((x.shape[0], 1))
            self.entrada = np.hstack((x, ones))
        else:
            self.entrada = x
            
        self.z = np.dot(self.entrada, self.Pesos)
        self.a = self.Activacion(self.z)
        return self.a
    
    def BackPropagation(self, lr: float, is_last_layer: bool, next_w: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if is_last_layer:
            error = y_true - self.a
            delta = error * self.transfer_derivative(self.a)
        else:
            delta = np.dot(y_true, next_w.T)
        
        gradiente_w = np.dot(self.entrada.T, delta)

        if self.optimizador is not None:
             self.Pesos = self.optimizador(self.Pesos, gradiente_w, lr)

        self.Pesos += lr * gradiente_w
        return delta

X = np.ones((3, 2)) * 5
y = np.zeros((3, 2))

capa = Layer(
    entradas = 2,
    neuronas = 2,
    activacion = "relu",
    bias = True,
    optimizador = None
)

# Setup
salida = capa.FeedForward(X)
print("Pesos iniciales:\n", capa.Pesos)

# Feedforward
salida = capa.FeedForward(X)
print("Salida (a):\n", salida)

# Backpropagation
tasa_aprendizaje = 0.01
capa.BackPropagation(lr = tasa_aprendizaje, is_last_layer = True, next_w = None, y_true = y)
print("Nuevos Pesos actualizados:\n", capa.Pesos)