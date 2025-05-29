import numpy as np
# Déterminer le type de matrice
def determine_matrix_type(matrice): #I don't know why it was: def determine_matrix_type(matrice, m=0)
    a = est_matrice_inf_bande(matrice)
    b = est_matrice_sup_bande(matrice)
    c = est_bande(matrice)
    d = np.allclose(matrice, np.diag(np.diagonal(matrice)))
    
    if np.all(np.array(matrice) == 0):
        return "nulle"
    elif np.array_equal(np.array(matrice), np.eye(len(matrice))):
        return "d'identité"
    elif d:
        return "Diagonale"
    elif a[0]:
        return "Demi-bande inférieure", a[1]
    elif b[0]:
        return "Demi-bande supérieure", b[1]
    elif c[0] and np.allclose(matrice, matrice.T):
        return "Bande symétrique définie positive", c[1]
    #elif c[0] and np.array_equal(matrice, matrice.T):
    #    return "Bande symétrique", c[1]
    elif c[0]:
        return "Bande", c[1]
    elif np.all(np.triu(matrice) == matrice):
        return "Triangulaire supérieure"
    elif np.all(np.tril(matrice) == matrice):
        return "Triangulaire inférieure"
    elif np.array_equal(matrice, matrice.T) and np.all(np.linalg.eigvals(matrice) > 0):
        return "Symétrique définie positive"
    elif np.array_equal(matrice, matrice.T):
        return "Symétrique"
    elif all(abs(matrice[i, i]) >= sum(abs(matrice[i, :])) - abs(matrice[i, i]) for i in range(matrice.shape[0])) and all(abs(matrice[i, i]) >= sum(abs(matrice[:, i])) - abs(matrice[i, i]) for i in range(matrice.shape[0])):
        return "Diagonale dominante"
    elif all(abs(matrice[i, i]) >= sum(abs(matrice[i, :])) - abs(matrice[i, i]) for i in range(matrice.shape[0])):
        return "Diagonale dominante par ligne"
    elif all(abs(matrice[i, i]) >= sum(abs(matrice[:, i])) - abs(matrice[i, i]) for i in range(matrice.shape[0])):
        return "Diagonale dominante par colonne"
    else:
        return "Dense"


def est_matrice_sup_bande(matrice):
    n = matrice.shape[0]
    if not np.all(np.triu(matrice) == matrice):
        return False,False
    # Compter le nombre de diagonales non nulles
    diagonales_nulles=diagonales_nulles_sup(matrice)
    if diagonales_nulles==0:
        return False,False
    return True,n-1-diagonales_nulles

def est_matrice_inf_bande(matrice):
    n = matrice.shape[0]
    # Vérifier si la matrice est une matrice triangulaire inférieure
    if not np.all(np.tril(matrice) == matrice):
        return False,False
    # Compter le nombre de diagonales non nulles
    diagonales_nulles =  diagonales_nulles_inf(matrice)
    if diagonales_nulles==0:
        return False,False
    return True,n-1-diagonales_nulles

def diagonales_nulles_inf(matrice):
    # Extraire les diagonales
    diagonales = [np.diag(matrice, k=i) for i in range(-matrice.shape[0]+1, 0)]
    print(diagonales)
    # Compter les zéros dans chaque diagonale
    diagonales_nulles_count = 0
    for diag in diagonales:
        if np.all(diag==0):
            diagonales_nulles_count+=1
        else:
            break
    return diagonales_nulles_count

def est_bande(matrice):
    matrice_inf = np.tril(matrice)
    matrice_sup = np.triu(matrice)
    a=est_matrice_inf_bande(matrice_inf)
    b=est_matrice_sup_bande(matrice_sup)
    if a[0] and b[0] :
        return True,max(a[1],b[1])
    else :
        return False,False


def diagonales_nulles_sup(matrice):
    # Extraire les diagonales
    diagonales = [np.diag(matrice, k=-i) for i in range(-matrice.shape[0]+1, 0)]
    print(diagonales)
    # Compter les zéros dans chaque diagonale
    diagonales_nulles_count = 0
    for diag in diagonales:
        if np.all(diag==0):
            diagonales_nulles_count+=1
        else:
            break
    return diagonales_nulles_count


def generate_specific_matrix(val_min, val_max, matrix_type, size, bande_width, demi_bande_width):
    if matrix_type == "Symétrique":
        A = np.random.randint(val_min, val_max, size=(size, size))
        A = (A + A.T) / 2  # Rendre la matrice symétrique
    elif matrix_type == "Dense":
        A = np.random.randint(val_min, val_max, size=(size, size))
    elif matrix_type == "Triangulaire supérieure":
        A = np.triu(np.random.randint(val_min, val_max, size=(size, size)))
    elif matrix_type == "Triangulaire inférieure":
        A = np.tril(np.random.randint(val_min, val_max, size=(size, size)))
    elif matrix_type == "Diagonale":
        A = np.diag(np.random.randint(val_min, val_max, size=(size,)))
    elif matrix_type == "Bande":
        A = np.zeros((size, size))
        for i in range(-bande_width, bande_width + 1):
            A += np.diag(np.random.randint(val_min, val_max, size=(size - abs(i))), k=i)
    elif matrix_type == "Demi-bande inférieure":
        A = np.zeros((size, size))
        for i in range(-demi_bande_width, 1):
            A += np.diag(np.random.randint(val_min, val_max, size=(size - abs(i))), k=i)
    elif matrix_type == "Demi-bande supérieure":
        A = np.zeros((size, size))
        for i in range(0, demi_bande_width + 1):
            A += np.diag(np.random.randint(val_min, val_max, size=(size - abs(i))), k=i)
    elif matrix_type == "Symétrique définie positive":
        A = np.random.randint(val_min, val_max, size=(size, size))
        A = np.dot(A, A.T) + np.eye(size)  # Rendre la matrice définie positive
    elif matrix_type == "Bande symétrique définie positive":
        A = np.zeros((size, size))
        for i in range(-bande_width, bande_width + 1):
            A += np.diag(np.random.randint(val_min, val_max, size=(size - abs(i))), k=i)
        A = (A + A.T) / 2 + np.eye(size)  # Symétrie et définie positive
    elif matrix_type == "Diagonale dominante":
        A = np.random.randint(val_min, val_max, size=(size, size))
        for i in range(size):
            # Calculer la somme des éléments hors diagonale pour la ligne et la colonne
            sum_row = sum(abs(A[i, :])) - abs(A[i, i])
            sum_col = sum(abs(A[:, i])) - abs(A[i, i])
            # Ajouter une valeur aléatoire pour assurer la stricte dominance dans les deux cas
            A[i, i] += np.random.randint(val_min, val_max) + max(sum_row, sum_col)
    else:
        raise ValueError("Type de matrice non pris en charge.")
    return A


def message_info(matrix_type):
    if matrix_type[0] in ["Bande", "Bande symétrique définie positive"]: 
        return f"Type de matrice A détecté : Matrice {matrix_type[0]} de largeur de bande m= {matrix_type[1]}"
    elif matrix_type[0] in ["Demi-bande inférieure", "Demi-bande supérieure"]:
        return f"Type de matrice A détecté : Matrice {matrix_type[0]} de largeur de demi bande m= {matrix_type[1]}"
    else:
        return f"Type de matrice A détecté : Matrice {matrix_type}"