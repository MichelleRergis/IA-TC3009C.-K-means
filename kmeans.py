"""

Implementacion de K-means con scikit-learn sobre un dataset sintetico.

Autor: Michelle Rergis Novelo
Materia: TC3006C - IA Modulo 2.

Ejecución:
    python kmeans_es.py
    python kmeans_es.py --k 4 --n-mue 1500 --n-car 6 --n-grp 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


# 1. Generación del dataset sintético
def generar_datos(n_mue: int = 1500, n_car: int = 6, n_grp: int = 4,
                   disp: float = 2.5, semilla: int = 42):
    """
    Genera un dataset sintético tipo blobs con grupos reales conocidos,
    útil para validar el desempeno de K-means contra una verdad base.
    """
    X_arr, y_arr = make_blobs(
        n_samples=n_mue,
        n_features=n_car,
        centers=n_grp,
        cluster_std=disp,
        random_state=semilla,
    )

    nom_car = [f"Var_{i+1}" for i in range(n_car)]
    nom_grp = [f"Grupo_{i}" for i in range(n_grp)]

    X = pd.DataFrame(X_arr, columns=nom_car)
    y = pd.Series(y_arr, name="Grupo")

    print("[INFO] Dataset sintetico generado con exito.")
    print(f"[INFO] Muestras: {n_mue} | Variables: {n_car} | Grupos reales: {n_grp}")

    return X, y, nom_car, nom_grp


# 2. Busqueda de K: método del codo + silueta

def buscar_k(X_e: np.ndarray, k_min: int, k_max: int, dir_sal: Path, semilla: int = 42):
    valores_k = list(range(k_min, k_max + 1))
    wcss = []
    sil = []

    for k in valores_k:
        mod = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=semilla)
        etq = mod.fit_predict(X_e)
        wcss.append(mod.inertia_)
        sil.append(silhouette_score(X_e, etq) if k > 1 else np.nan)

    # Gráfica del codo
    plt.figure(figsize=(7, 5))
    plt.plot(valores_k, wcss, marker="o", color="b")
    plt.title("Metodo del Codo (Elbow Method)")
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("WCSS (Inercia)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dir_sal / "01_metodo_codo.png", dpi=150)
    plt.close()

    # Gráfica de silueta
    plt.figure(figsize=(7, 5))
    k_val = [k for k in valores_k if k > 1]
    sil_val = [s for k, s in zip(valores_k, sil) if k > 1]
    plt.plot(k_val, sil_val, marker="o", color="darkorange")
    plt.title("Coeficiente de Silueta por numero de clusters")
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dir_sal / "02_silueta.png", dpi=150)
    plt.close()

    k_mejor = k_val[int(np.argmax(sil_val))]
    print(f"[INFO] K sugerido segun silhouette score: {k_mejor}")
    return k_mejor


# 3. Mapeo cluster -> grupo real (voto mayoritario)

def mapear_grupos(etq_pred: np.ndarray, etq_real: np.ndarray) -> dict:
    mapa = {}
    for c in np.unique(etq_pred):
        mask = etq_pred == c
        if mask.sum() == 0:
            continue
        mapa[c] = pd.Series(etq_real[mask]).mode()[0]
    return mapa


# 4. Visualización (proyeccion PCA a 2D)
def graficar_clusters(X_e: np.ndarray, etq: np.ndarray, centros: np.ndarray,
                       titulo: str, archivo: Path):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_e)
    centros_2d = pca.transform(centros)

    plt.figure(figsize=(8, 6))
    disp = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=etq, cmap="tab10",
                        s=25, alpha=0.6, edgecolor="none")
    plt.scatter(centros_2d[:, 0], centros_2d[:, 1], c="red", marker="X", s=200,
                edgecolor="black", linewidth=1.5, label="Centroides")
    plt.title(titulo)
    plt.xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    plt.ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    plt.legend()
    plt.colorbar(disp, label="Cluster")
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


def graficar_matriz(cm: np.ndarray, nom_grp: list, archivo: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nom_grp, yticklabels=nom_grp)
    plt.title("Matriz de Confusion (cluster mapeado vs. grupo real)")
    plt.xlabel("Prediccion (cluster mapeado)")
    plt.ylabel("Grupo real")
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


# 4b. Curva de convergencia (equivalente a "training/loss curve" en K-means)
def curva_convergencia(X_e: np.ndarray, k: int, semilla: int, dir_sal: Path,
                        max_iter: int = 25, tol: float = 1e-6):
    """
    K-means no tiene una funcion de perdida por epocas como una red neuronal,
    pero si tiene una funcion objetivo (la inercia / WCSS) que disminuye de
    forma monotona en cada iteracion hasta la convergencia. Esta funcion
    reproduce ese comportamiento paso a paso: se inicializan centroides con
    k-means++ y se ejecuta UNA iteracion de K-means a la vez (max_iter=1),
    usando los centroides resultantes como punto de partida de la siguiente,
    registrando la inercia en cada paso. Esto genera una curva de
    convergencia analoga a una curva de entrenamiento/perdida.
    """
    centros, _ = kmeans_plusplus(X_e, n_clusters=k, random_state=semilla)
    inercias = []

    for it in range(1, max_iter + 1):
        mod = KMeans(n_clusters=k, init=centros, n_init=1, max_iter=1, random_state=semilla)
        mod.fit(X_e)
        inercias.append(mod.inertia_)
        centros_nuevos = mod.cluster_centers_
        cambio = np.linalg.norm(centros_nuevos - centros)
        centros = centros_nuevos
        if cambio < tol:
            break

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(inercias) + 1), inercias, marker="o", color="green")
    plt.title(f"Curva de Convergencia de K-means (K={k})")
    plt.xlabel("Iteracion")
    plt.ylabel("Inercia (WCSS) - funcion objetivo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dir_sal / "06_curva_convergencia.png", dpi=150)
    plt.close()

    return inercias


# 4c. Exportar clusters de entrenamiento y prueba a CSV
def guardar_clusters(X: pd.DataFrame, y_real: pd.Series, etq: np.ndarray,
                      mapa: dict, nom_grp: list, archivo: Path):
    """Guarda un CSV con las variables originales, el grupo real, el cluster
    asignado por K-means y el grupo predicho (segun el mapeo por voto mayoritario)."""
    df = X.reset_index(drop=True).copy()
    y_real = y_real.reset_index(drop=True)
    df["grupo_real_id"] = y_real
    df["grupo_real_nombre"] = y_real.apply(lambda i: nom_grp[i])
    df["cluster_asignado"] = etq
    df["grupo_predicho"] = [
        nom_grp[mapa[c]] if c in mapa and 0 <= mapa[c] < len(nom_grp) else "Desconocido"
        for c in etq
    ]
    df.to_csv(archivo, index=False, encoding="utf-8")
    print(f"[INFO] Clusters guardados en: {archivo} ({len(df)} filas)")


# 5. Predicción de un punto nuevo (uso en consola)
def predecir_punto(modelo: KMeans, esc: StandardScaler, mapa: dict,
                    nom_grp: list, nom_car: list, valores: list) -> None:
    #Recibe una lista de valores crudos (una fila) y muestra el cluster asignado.
    punto = pd.DataFrame([valores], columns=nom_car)
    punto_e = esc.transform(punto)
    cluster = modelo.predict(punto_e)[0]
    grupo = mapa.get(cluster, -1)
    nombre = nom_grp[grupo] if 0 <= grupo < len(nom_grp) else "Desconocido"
    print(f"[PREDICCION] Punto {valores} -> Cluster {cluster} -> {nombre}")


# 6. Main
def main():
    ap = argparse.ArgumentParser(description="K-means con dataset sintetico (scikit-learn)")
    ap.add_argument("--n-mue", type=int, default=1500, help="Numero de muestras a generar")
    ap.add_argument("--n-car", type=int, default=6, help="Numero de variables (features)")
    ap.add_argument("--n-grp", type=int, default=4, help="Numero de grupos reales a generar")
    ap.add_argument("--disp", type=float, default=2.5, help="Dispersion (cluster_std) de los grupos")
    ap.add_argument("--k", type=int, default=4,
                     help="K fijo para el modelo final (por defecto 4). Usa 0 para K automatico (metodo de silueta).")
    ap.add_argument("--k-rango", type=int, nargs=2, default=[2, 8], metavar=("K_MIN", "K_MAX"))
    ap.add_argument("--prueba", type=float, default=0.2, help="Proporcion para el set de prueba")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--dir-sal", type=str, default="salida")
    args = ap.parse_args()

    dir_sal = Path(args.dir_sal)
    dir_sal.mkdir(parents=True, exist_ok=True)

    lineas = []

    def log(msg: str):
        print(msg)
        lineas.append(msg)

    log("=" * 70)
    log("K-MEANS - REPORTE DE EJECUCION (DATASET SINTETICO)")
    log("=" * 70)

    # 1. Generar datos 
    X, y, nom_car, nom_grp = generar_datos(
        n_mue=args.n_mue, n_car=args.n_car, n_grp=args.n_grp, disp=args.disp, semilla=args.semilla
    )

    # 2. División entrenamiento / prueba 
    X_ent, X_prb, y_ent, y_prb = train_test_split(
        X, y, test_size=args.prueba, random_state=args.semilla, stratify=y
    )
    log(f"[SPLIT] Entrenamiento: {X_ent.shape[0]} muestras | Prueba: {X_prb.shape[0]} muestras")

    # 3. Escalado 
    esc = StandardScaler()
    X_ent_e = esc.fit_transform(X_ent)
    X_prb_e = esc.transform(X_prb)
    log("[PREP] Variables escaladas con StandardScaler (media 0, desv. 1).")

    # 4. Busqueda de K 
    k_min, k_max = args.k_rango
    k_mejor = buscar_k(X_ent_e, k_min, k_max, dir_sal, semilla=args.semilla)
    k_final = args.k if args.k and args.k > 0 else k_mejor
    log(f"[K] Valor de K utilizado en el modelo final: {k_final}")

    # 5. Entrenamiento del modelo final 
    modelo = KMeans(n_clusters=k_final, init="k-means++", n_init=10,
                     max_iter=300, random_state=args.semilla)
    modelo.fit(X_ent_e)

    grp_ent = modelo.labels_
    grp_prb = modelo.predict(X_prb_e)

    log(f"[MODELO] Inercia final (entrenamiento): {modelo.inertia_:.3f}")
    log(f"[METRICA] Silhouette (entrenamiento): {silhouette_score(X_ent_e, grp_ent):.4f}")
    log(f"[METRICA] Silhouette (prueba): {silhouette_score(X_prb_e, grp_prb):.4f}")

    # 5b. Distribución de clusters (entrenamiento y prueba) 
    log("[CLUSTERS] Distribucion de clusters en ENTRENAMIENTO:")
    for c, n in pd.Series(grp_ent).value_counts().sort_index().items():
        log(f"    Cluster {c}: {n} muestras")
    log("[CLUSTERS] Distribucion de clusters en PRUEBA:")
    for c, n in pd.Series(grp_prb).value_counts().sort_index().items():
        log(f"    Cluster {c}: {n} muestras")

    # 5c. Curva de convergencia (equivalente a curva de entrenamiento/perdida) 
    inercias_iter = curva_convergencia(X_ent_e, k_final, args.semilla, dir_sal)
    log(f"[CONVERGENCIA] Inercia inicial (iter. 1): {inercias_iter[0]:.3f}")
    log(f"[CONVERGENCIA] Inercia final (iter. {len(inercias_iter)}): {inercias_iter[-1]:.3f}")
    log(f"[CONVERGENCIA] El algoritmo convergio en {len(inercias_iter)} iteraciones.")

    #  6. Visualización 
    graficar_clusters(X_ent_e, grp_ent, modelo.cluster_centers_,
                       f"Clusters en Entrenamiento (K={k_final})", dir_sal / "03_clusters_entrenamiento.png")
    graficar_clusters(X_prb_e, grp_prb, modelo.cluster_centers_,
                       f"Clusters en Prueba (K={k_final})", dir_sal / "04_clusters_prueba.png")

    #  7. Evaluación contra la verdad base 
    ari = adjusted_rand_score(y_prb, grp_prb)
    nmi = normalized_mutual_info_score(y_prb, grp_prb)
    log(f"[METRICA] Adjusted Rand Index (ARI): {ari:.4f}")
    log(f"[METRICA] Normalized Mutual Info (NMI): {nmi:.4f}")

    mapa = mapear_grupos(grp_ent, y_ent.to_numpy())
    y_pred = np.array([mapa.get(c, -1) for c in grp_prb])

    cm = confusion_matrix(y_prb, y_pred)
    acc = accuracy_score(y_prb, y_pred)
    f1 = f1_score(y_prb, y_pred, average="macro", zero_division=0)

    log(f"[EVAL] Mapeo cluster->grupo: {mapa}")
    log(f"[EVAL] Accuracy: {acc:.4f} | F1-Score (macro): {f1:.4f}")

    graficar_matriz(cm, nom_grp, dir_sal / "05_matriz_confusion.png")

    #  7b. Exportar clusters de entrenamiento y prueba (K final) 
    guardar_clusters(X_ent, y_ent, grp_ent, mapa, nom_grp, dir_sal / "clusters_entrenamiento.csv")
    guardar_clusters(X_prb, y_prb, grp_prb, mapa, nom_grp, dir_sal / "clusters_prueba.csv")

    #  8. Predicciones de ejemplo en consola 
    log("-" * 70)
    log("[DEMO] Prediccion de puntos nuevos (consola):")
    for i in range(3):
        fila = X_prb.iloc[i].tolist()
        predecir_punto(modelo, esc, mapa, nom_grp, nom_car, fila)
        lineas.append(f"[DEMO] Punto {fila} evaluado.")

    #  9. Guardar reporte de texto 
    ruta_txt = dir_sal / "resultados.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    log(f"[OK] Proceso terminado. Resultados y graficas en: {dir_sal.resolve()}")


if __name__ == "__main__":
    sys.exit(main())