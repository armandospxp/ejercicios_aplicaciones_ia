""""

PROYECTO 2  Aplicaciones de IA

Aprendizaje No Supervisado


Nombre: Armando Rodriguez


email: armandospxp@gmail.com


"""
# Importacion de librerias

# pandas y matplotplib
import pandas as pd
import matplotlib.pyplot as plt

# dataset
from sklearn.datasets import load_iris

# modelos ml
from sklearn.cluster import KMeans, AgglomerativeClustering

# standarizacion numerica y PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# metricas
from sklearn.metrics import silhouette_score, davies_bouldin_score


def carga_datos() -> pd.DataFrame:
    """Metodo que retorna 2 dataframes con la data necesaria para el entrenamiento"""
    iris = load_iris()
    return pd.DataFrame(iris.data, columns=iris.feature_names)


def estandarizacion(X: pd.DataFrame) -> pd.Series:
    """Metodo para estandarizar los datos con StandarScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


if __name__ == '__main__':
    """Corrida del script"""
    print("Inicio del Script")
    data = carga_datos()

    # Shape y nombre de columnas

    print(f"Dataset shape: {data.shape}")
    print(f"Nombre de columnas: {data.columns}")

    # estandarizacion de la data

    d_est = estandarizacion(data)

    # aplicar PCA en 2 dimensiones

    pca = PCA(n_components=2, random_state=42)
    d_pca = pca.fit_transform(d_est)
    print(
        f"Ratio de varianza para 2 componentess): {pca.explained_variance_ratio_.sum():.2%}")

    # ajustamos el numero de clusters a 3
    n_clusters = 3

    # Algoritmo KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiquetas_kmeans = kmeans.fit_predict(d_est)

    # Algoritmo Agglomerative Clustering

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    etiquetas_agg = agg.fit_predict(d_est)

    # 4. Evaluate clustering quality
    metrics = {
        "Algoritmo": [],
        "Silhouette": [],
        "Davies-Bouldin": []
    }
    print("\n=== Evaluación de la calidad del clustering ===")
    for nombre, etiquetas in [("Algoritmo KMeans", etiquetas_kmeans), ("Algorigtmo Agglomerative", etiquetas_agg)]:
        sil = silhouette_score(d_est, etiquetas)
        db = davies_bouldin_score(d_est, etiquetas)
        metrics["Algoritmo"].append(nombre)
        metrics["Silhouette"].append(sil)
        metrics["Davies-Bouldin"].append(db)

    print("\n=== Métricas de clustering ===")
    for i in range(len(metrics["Algoritmo"])):
        print(f"{metrics['Algoritmo'][i]:>15}: Silhouette = {metrics['Silhouette'][i]:.3f}, "
              f"Davies-Bouldin = {metrics['Davies-Bouldin'][i]:.3f}")

    # 5. Visualizar clusteres 2D
    print("\n=== Visualización de los clusters en 2D ===")
    # KMeans
    plt.figure()
    plt.scatter(d_pca[:, 0], d_pca[:, 1], c=etiquetas_kmeans)
    plt.title("Clusteres para Kmeans (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.show()

    # Agglomerative
    plt.figure()
    plt.scatter(d_pca[:, 0], d_pca[:, 1], c=etiquetas_agg)
    plt.title("Clusters para Agglomerative (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.show()
