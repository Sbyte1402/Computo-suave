import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONES DE MEMBRESÍA (MFs)
# ==========================================
def trimf(x, a, b, c):
    """Función de membresía triangular"""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal"""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))

def gaussmf(x, mean, sigma):
    """Función de membresía gaussiana"""
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# --- CREACIÓN DE LAS 3 VARIABLES LINGÜÍSTICAS ---
# Universo de discurso para las variables
x_temp = np.linspace(0, 40, 100)   # Variable 1: Temperatura (0 a 40 °C)
x_vel = np.linspace(0, 120, 100)   # Variable 2: Velocidad (0 a 120 km/h)
x_pres = np.linspace(0, 100, 100)  # Variable 3: Presión (0 a 100 psi)

# Variable 1: Temperatura (Mezcla: Trapecio, Triángulo, Trapecio)
temp_fria = trapmf(x_temp, 0, 0, 10, 20)
temp_templada = trimf(x_temp, 10, 20, 30)
temp_caliente = trapmf(x_temp, 20, 30, 40, 40)

# Variable 2: Velocidad (Mezcla: Gaussiana, Gaussiana, Gaussiana)
vel_lenta = gaussmf(x_vel, 20, 15)
vel_media = gaussmf(x_vel, 60, 15)
vel_rapida = gaussmf(x_vel, 100, 15)

# Variable 3: Presión (Mezcla: Trapecio, Gaussiana, Triángulo)
pres_baja = trapmf(x_pres, 0, 0, 20, 40)
pres_normal = gaussmf(x_pres, 50, 10)
pres_alta = trimf(x_pres, 60, 80, 100)


# ==========================================
# 2. OPERADORES DE CONJUNTOS DIFUSOS TIPO 1
# ==========================================
def fuzzy_union(A, B):
    """Unión difusa (S-Norma: Máximo)"""
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    """Intersección difusa (T-Norma: Mínimo)"""
    return np.minimum(A, B)

def fuzzy_complement(A):
    """Complemento difuso"""
    return 1.0 - A


# ==========================================
# 3. PROCESO DE COMPOSICIÓN (Max-Min)
# ==========================================
def composicion_max_min(R, S):
    """Composición Max-Min de dos relaciones R y S"""
    R = np.array(R)
    S = np.array(S)
    m, n = R.shape
    n_s, p = S.shape
    
    if n != n_s:
        raise ValueError("Las dimensiones de las matrices no coinciden para la composición")
        
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            # Maximo de los minimos entre la fila i de R y la columna j de S
            C[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
    return C

# Comprobación con las matrices R y S solicitadas
R = [[1, 0], 
     [0, 1]]

S = [[1, 0, 0], 
     [1, 0, 1]]

resultado_max_min = composicion_max_min(R, S)
print("=== Resultado de Composición Max-Min (R o S) ===")
print("Matriz R:\n", np.array(R))
print("Matriz S:\n", np.array(S))
print(resultado_max_min)


# ==========================================
# 4. COMPOSICIÓN CON T-NORMA = PRODUCTO (Max-Producto)
# ==========================================
def composicion_max_producto(R, S):
    """Composición Max-Producto de dos relaciones R y S"""
    R = np.array(R)
    S = np.array(S)
    m, n = R.shape
    n_s, p = S.shape
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            # Maximo de los PRODUCTOS entre la fila i de R y la columna j de S
            C[i, j] = np.max(R[i, :] * S[:, j])
    return C

R = [[1, 0], 
     [0, 1]]

S = [[1, 0, 0], 
     [1, 0, 1]]

resultado_max_producto = composicion_max_producto(R, S)
print("=== Resultado de Composición Max-Producto (R o S) ===")
print("Matriz R:\n", np.array(R))
print("Matriz S:\n", np.array(S))
print(resultado_max_producto)