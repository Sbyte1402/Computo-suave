import numpy as np
import time as tm
import pandas as pd
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
computer_hardware = fetch_ucirepo(id=29) 
wine_quality = fetch_ucirepo(id=186)
student_performance = fetch_ucirepo(id=320)
  
class Standardizer:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0
        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X * self.std) + self.mean

class Layer():
    def __init__(self, entradas: int, neuronas: int, activacion: str, bias: bool, optimizador = None, dropout_rate: float = 0.0):
        self.Neuronas = neuronas
        self.Bias = bias
        self.optimizador = optimizador
        self.funcion_activacion = activacion.lower()
        self.dropout_rate = dropout_rate
        self.keep_prob = 1.0 - dropout_rate
        self.mask = None
        
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
        elif nombre == "linear":
            self.Activacion = lambda x: x
        else:
            raise ValueError(f"Activación {nombre} no soportada.")

    def derivada_activacion(self) -> np.ndarray:
        if self.funcion_activacion == "sigmoid":
            return self.a_raw * (1 - self.a_raw)
        elif self.funcion_activacion == "relu":
            return np.where(self.z > 0, 1.0, 0.0)
        elif self.funcion_activacion == "leaky_relu":
            return np.where(self.z > 0, 1.0, 0.01)
        elif self.funcion_activacion in ["linear", "softmax"]:
            return np.ones_like(self.a_raw)
        return np.ones_like(self.a_raw)

    def FeedForward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if self.Bias:
            ones = np.ones((x.shape[0], 1))
            self.entrada = np.hstack((x, ones))
        else:
            self.entrada = x
            
        self.z = np.dot(self.entrada, self.Pesos)
        self.a_raw = self.Activacion(self.z)
        
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*self.a_raw.shape) < self.keep_prob) / self.keep_prob
            self.a = self.a_raw * self.mask
        else:
            self.mask = None
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
        
        if self.mask is not None:
            self.delta = self.delta * self.mask
            
        self.gradiente_w = np.dot(self.entrada.T, self.delta) / m

        if self.optimizador is not None:
             self.Pesos = self.optimizador(self.Pesos, self.gradiente_w, lr)
        else:
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
            output = layer.FeedForward(output, training=False)
        return output

    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 2000, lr: float = 0.01, 
            performance_goal: float = 1e-4, grad_goal: float = 1e-5, print_every: int = 500):
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        inicio = tm.time()
        print("Entrenamiento iniciado...")

        for epoch in range(max_epochs + 1):
            
            out_train = X
            # 1. Feedforward
            for layer in self.layers:
                out_train = layer.FeedForward(out_train, training=True)

            yHat_eval = self.predict(X)
            perf = np.mean((yHat_eval - y) ** 2)

            # 2. Backpropagation
            delta = None
            for i in reversed(range(len(self.layers))):
                is_last = (i == len(self.layers) - 1)
                next_w = self.layers[i + 1].Pesos if not is_last else None
                
                delta = self.layers[i].BackPropagation(
                    lr=lr,
                    is_last_layer=is_last,
                    next_w=next_w,
                    delta_siguiente=delta,
                    y_true=y
                )
            
            grad_norm = np.sqrt(sum(np.sum(capa.gradiente_w ** 2) for capa in self.layers))

            if epoch % print_every == 0 or epoch == 0 or epoch == max_epochs:
                print(f"Epoch {epoch:5d} / {max_epochs} , Cost {perf:.3e} / {performance_goal:.0e} , Gradient {grad_norm:.3e} / {grad_goal:.0e}")

            if perf <= performance_goal or grad_norm <= grad_goal:
                print("\nRedNeuronal : Performance or gradient goal met\n")
                break
        else:
            print("\nRedNeuronal : Maximum epoch reached, performance goal was not met\n")

        fin = tm.time()
        duracion = fin - inicio
        totalTime = duracion if duracion < 60 else (duracion / 60)
        unidad = "segundos" if duracion < 60 else "minutos"

        print(f"Tiempo total de entrenamiento: {totalTime:.2f} {unidad}\n")

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))

def mse_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mae_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true.ravel() == y_pred.ravel()))

def precision_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f, y_pred_f = y_true.ravel(), y_pred.ravel()
    tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
    fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
    return float(tp / (tp + fp + 1e-8))

def roc_auc_score_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true_f, y_scores_f = y_true.ravel(), y_scores.ravel()
    pos_mask = (y_true_f == 1)
    n_pos = np.sum(pos_mask)
    n_neg = len(y_true_f) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_scores_f)) + 1
    pos_rank_sum = np.sum(ranks[pos_mask])
    return float((pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg))

# ----------------------------------------------------------------------------------------------------
def cargar_y_preprocesar(dataset_id: int):
    dataset = fetch_ucirepo(id=dataset_id)
    X_raw = pd.DataFrame(dataset.data.features)
    y_raw = pd.DataFrame(dataset.data.targets)

    if y_raw.empty or y_raw.shape[1] == 0:
        target_col = X_raw.columns[-1]
        y_vec = X_raw[target_col].values
        X_raw = X_raw.drop(columns=[target_col])
    else:
        y_vec = y_raw.iloc[:, -1].values

    cols_a_eliminar = [
        col for col in X_raw.select_dtypes(include=['object', 'string']).columns
        if X_raw[col].nunique() > 50
    ]
    if cols_a_eliminar:
        X_raw = X_raw.drop(columns=cols_a_eliminar)

    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=float)
    X_encoded = X_encoded.fillna(X_encoded.mean())

    return X_encoded.values.astype(np.float32), y_vec.reshape(-1, 1).astype(np.float32), dataset.metadata.name

if __name__ == "__main__":
    datasets_ids = [29, 186, 320]  # Computer Hardware (29), Wine Quality (186), Student Performance (320)

    for id_ds in datasets_ids:
        X, y, nombre_ds = cargar_y_preprocesar(id_ds)
        
        print("=" * 65)
        print(f"Dataset: {nombre_ds} (ID {id_ds})")
        print(f"Dimensiones: {X.shape[0]} muestras, {X.shape[1]} características")
        print("=" * 65)

        # División Entrenamiento / Prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

        scaler_X = Standardizer()
        scaler_y = Standardizer()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        num_features = X_train_scaled.shape[1]
        
        red = NeuralNetwork()
        red.add_layer(Layer(entradas=num_features, neuronas=32, activacion="relu", bias=True, dropout_rate=0.0))
        red.add_layer(Layer(entradas=32, neuronas=16, activacion="relu", bias=True, dropout_rate=0.0))
        red.add_layer(Layer(entradas=16, neuronas=1, activacion="linear", bias=True))

        red.fit(
            X_train_scaled, 
            y_train_scaled, 
            max_epochs=20000, 
            lr=0.01, 
            performance_goal=1e-4, 
            grad_goal=1e-5, 
            print_every=1000
        )

        predicciones_scaled = red.predict(X_test_scaled)
        predicciones = scaler_y.inverse_transform(predicciones_scaled)
        
       # Evaluación con Métricas de Regresión
        r2 = r2_score_np(y_test, predicciones)
        mse = mse_score_np(y_test, predicciones)
        mae = mae_score_np(y_test, predicciones)

        print(f"--- Evaluación Final Regresión ({nombre_ds}) ---")
        print(f"• R^2 Score : {r2:.4f}")
        print(f"• MSE       : {mse:.4f}")
        print(f"• MAE       : {mae:.4f}")

        # Evaluación con Métricas de Clasificación Binaria (Binarizando la mediana)
        mediana_y = np.median(y_test)
        y_test_bin = (y_test > mediana_y).astype(int)
        pred_bin = (predicciones > mediana_y).astype(int)
    

        acc = accuracy_score_np(y_test_bin, pred_bin)
        prec = precision_score_np(y_test_bin, pred_bin)
        auc = roc_auc_score_np(y_test_bin, predicciones)

        print(f"--- Métricas Binarizadas (> Mediana {mediana_y:.1f}) ---")
        print(f"• Accuracy  : {acc:.4f}")
        print(f"• Precision : {prec:.4f}")
        print(f"• ROC-AUC   : {auc:.4f}\n\n")