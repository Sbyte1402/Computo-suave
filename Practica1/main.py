import numpy as np

class Layer():
    def __init__(self, entradas: int, neuronas: int, activacion: str, bias: bool):
        self.Neuronas = neuronas
        self.Bias = bias
        
        filas = entradas + 1 if bias else entradas
        self.Pesos = 0

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

    def FeedForward(self, x: np.ndarray) -> np.ndarray:
        self.z = np.dot(x, self.Pesos)
        self.a = self.Activacion(self.z)
        return self.a


x = np.array([[1, 2, 3],
              [4, 5, 6]])

w = np.array([[0  ,   0, 0.1, 0.2],
              [0.3,  -1, 0.4, 0.5],
              [1  , 0.6, 0.7, 0.8]])

capa = Layer(
    entradas = 3,
    neuronas = 4,
    activacion = "relu",
    bias = True
)

capa.Pesos = w
salida = capa.FeedForward(x)
print("\nSalida FeedForward:\n", salida)