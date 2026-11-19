
import numpy as np


# A. ACP : variance maximale et erreur de reconstruction

def demonstration_acp(X, q):
    """
    Calcule les q premières composantes principales
    et l'erreur de reconstruction.
    X doit être centré.
    """

    S = X.T @ X

    eigenvalues, eigenvectors = np.linalg.eigh(S)

    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    V = eigenvectors[:, :q]

    Z = X @ V
    X_reconstructed = Z @ V.T

    erreur = np.linalg.norm(X - X_reconstructed, ord="fro")**2

    return V, eigenvalues, erreur


# B. SVD

def demonstration_svd(X):
    """
    Décomposition SVD et vérification de X.T @ X = V @ Sigma² @ V.T.
    """

    U, singular_values, Vt = np.linalg.svd(
        X, full_matrices=False
    )

    Sigma_squared = np.diag(singular_values**2)

    XTX = X.T @ X
    reconstruction = Vt.T @ Sigma_squared @ Vt

    erreur = np.linalg.norm(XTX - reconstruction, ord="fro")

    return U, singular_values, Vt, erreur


# C. Variance expliquée

def variance_expliquee(eigenvalues, dimensions=[2, 5, 10, 20, 40]):
    """
    Calcule la variance expliquée cumulée.
    """

    total = np.sum(eigenvalues)
    resultats = {}

    for q in dimensions:
        ve = np.sum(eigenvalues[:q]) / total
        resultats[q] = ve * 100

    return resultats


# D. Projection gaussienne

def projection_gaussienne(X, q, seed=42):
    """
    Génère une matrice gaussienne et projette X.
    """

    rng = np.random.default_rng(seed)

    d = X.shape[1]

    R = rng.normal(0, 1 / np.sqrt(q), size=(d, q))

    Z = X @ R

    return Z, R


def verifier_norme_gaussienne(u, q=10, repetitions=1000, seed=42):
    """
    Vérifie expérimentalement E[||R.T @ u||²] = ||u||².
    """

    rng = np.random.default_rng(seed)

    d = len(u)
    normes = []

    for _ in range(repetitions):
        R = rng.normal(0, 1 / np.sqrt(q), size=(d, q))

        norme_projetee = np.linalg.norm(R.T @ u)**2
        normes.append(norme_projetee)

    moyenne = np.mean(normes)
    norme_originale = np.linalg.norm(u)**2

    return norme_originale, moyenne


# E. Johnson-Lindenstrauss

def borne_dimension_jl(n, epsilon, delta=0.05):
    """
    Estimation illustrative de la dimension suffisante :
    q >= 8 * epsilon^(-2) * log(n/delta).

    La constante 8 est un choix conservateur illustratif,
    pas une constante universelle optimale.
    """

    q = 8 * (epsilon**-2) * np.log(n / delta)

    return int(np.ceil(q))

