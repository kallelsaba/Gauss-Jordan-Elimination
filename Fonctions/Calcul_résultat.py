import numpy as np
from fractions import Fraction

# Gauss-Jordan pour matrices denses
def gauss_jordan_dense(A, b, n, afficher_details=False):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    matrice_aug = np.hstack([A, b])
    iterations = []
    for k in range(n):
        #pivotage partiel
        # Trouver le pivot maximal
        max_pivot_index = np.argmax(np.abs(matrice_aug[k:, k])) + k
        # Vérifier que que le pivot n'est pas null
        if matrice_aug[max_pivot_index, k] == 0:
            raise ValueError("La matrice est singulière ou mal conditionnée.")
        #échanger les lignes
        matrice_aug[[k, max_pivot_index], :] = matrice_aug[[max_pivot_index, k], :]

        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        matrice_aug[k, :] = matrice_aug[k, :]/pivot
        # Éliminer les autres lignes
        for i in range(n):
            if i != k:
                matrice_aug[i, :] = matrice_aug[i, :]-matrice_aug[i, k] * matrice_aug[k, :]
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations


#la fonction gauss_jordan_sans_pivotage est utlisée dans le cas d'une matrice diagonale dominante par ligne(ou|et) par colonne et dans le cas d'une matrice symétrique
def gauss_jordan_sans_pivotage(A, b, n, afficher_details=False):
    # Créer la matrice augmentée [A | b]
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    matrice_aug = np.hstack([A, b])

    # Liste pour stocker les itérations si afficher_details est True
    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
            
            # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot

        # Élimination des autres éléments dans la colonne k
        for i in range(n):
            if i != k:
                matrice_aug[i, :] = matrice_aug[i, :] - matrice_aug[i, k] * matrice_aug[k, :]

        # Sauvegarder l'état actuel de la matrice augmentée si afficher_details est True
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations



def gauss_jordan_diagonale(A,b,n,afficher_details=False):
    A = np.array(A, dtype=object)  # Utiliser dtype=object pour manipuler des fractions
    b = np.array(b, dtype=object)
    
    # Créer la matrice augmentée (A | b)
    matrice_aug = np.hstack((A, b.reshape(-1, 1)))

    # Liste pour stocker les étapes
    iterations = []

    for k in range(n):
        # Normalisation de la ligne par le pivot
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
        
        matrice_aug[k, :] = matrice_aug[k, :] / pivot
        matrice_aug[k, k]=1
        # Enregistrer l'état actuel de la matrice augmentée si afficher_details est True
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations


# Gauss-Jordan pour matrices triangulaires supérieures
def gauss_jordan_triangulaire_sup(A, b, n, afficher_details=False):
    A = np.array(A, dtype=float)
    b = np.squeeze(np.array(b, dtype=float)) # Assurez que b est 1D
    matrice_aug = np.hstack((A, b.reshape(-1, 1)))

    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    if not resultat:
        for k in range(n-1,-1,-1):
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
            
            # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot

            for j in range(k-1,-1,-1):
                # aug_matrix[i-1,:]=aug_matrix[i-1,:]-aug_matrix[i,:]*aug_matrix[i-1,i]
                matrice_aug[j,n]=matrice_aug[j,n]-matrice_aug[j,k]*matrice_aug[k,n]
                matrice_aug[j,k]=0
            
            # Sauvegarder l'étape si nécessaire
            if afficher_details:
                iterations.append(matrice_aug.copy())

            

       # Sauvegarder l'étape si nécessaire
        #if afficher_details:
        #    iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:,-1]
    return x, iterations


# Gauss-Jordan pour matrices triangulaires inférieures
def gauss_jordan_triangulaire_inf(A, b, n, afficher_details=False):
    A = np.array(A, dtype=float)
    b = np.squeeze(np.array(b, dtype=float)) # Assurez que b est 1D
    matrice_aug = np.hstack((A, b.reshape(-1, 1)))

    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
                
                # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot
            #if afficher_details:
            #    iterations.append(matrice_aug.copy())

        for i in range(k+1,n):
            # aug_matrix[i-1,:]=aug_matrix[i-1,:]-aug_matrix[i,:]*aug_matrix[i-1,i]
            matrice_aug[i,n]=matrice_aug[i,n]-matrice_aug[i,k]*matrice_aug[k,n]
            matrice_aug[i,k]=0

        # Sauvegarder l'étape si nécessaire
        if afficher_details:
                iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations

# Gauss-Jordan pour matrices bande
def gauss_jordan_bande(A, b, n, m, afficher_details=False): 
    A = np.array(A, dtype=float)
    b = np.squeeze(np.array(b, dtype=float)).reshape(-1, 1)
    matrice_aug = np.hstack([A, b])  # Construire la matrice augmentée
    iterations = []

    for k in range(n):  # Parcours des lignes pour Gauss-Jordan
        #pivotage partiel
        # Trouver le pivot maximal
        max_pivot_index = np.argmax(np.abs(matrice_aug[k:, k])) + k
        # Vérifier que que le pivot n'est pas null
        if matrice_aug[max_pivot_index, k] == 0:
            raise ValueError("La matrice est singulière ou mal conditionnée.")
        #échanger les lignes
        matrice_aug[[k, max_pivot_index], :] = matrice_aug[[max_pivot_index, k], :]

        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        matrice_aug[k, :] = matrice_aug[k, :]/pivot

        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m + 1, n)): 
            matrice_aug[i, :]=matrice_aug[i, :]-matrice_aug[k, :]*matrice_aug[i,k]
        
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
            iterations.append(matrice_aug.copy())
        
    # Élimination au dessus de la diagonale 
    for i in range(n-1,0,-1):
        for j in range(i-1,-1,-1):
            matrice_aug[j,n]=matrice_aug[j,n]-matrice_aug[j,i]*matrice_aug[i,n]
            matrice_aug[j,i]=0
    
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
                iterations.append(matrice_aug.copy())
    

    # Extraction de la solution
    x = matrice_aug[:, -1]
    return x, iterations


# Gauss-Jordan pour matrices demi-bande inférieures 
def gauss_jordan_demi_bande_inf(A, b, n, m, afficher_details=False):
    A = np.array(A, dtype=float)
    b = np.squeeze(np.array(b, dtype=float)).reshape(-1, 1)
    matrice_aug = np.hstack([A, b])
    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):  # Parcours des lignes pour Gauss-Jordan
        # Normaliser la ligne k
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
            matrice_aug[k, :] = matrice_aug[k, :]/pivot

        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m + 1, n)): 
            matrice_aug[i, n]=matrice_aug[i, n]-matrice_aug[k, n]*matrice_aug[i,k]
            matrice_aug[i,k]=0


        # Enregistrer les détails si demandé
        if afficher_details:
            iterations.append(matrice_aug.copy())

    # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations

# Gauss-Jordan pour matrices demi-bande supérieures 
def gauss_jordan_demi_bande_sup(A, b, n, m, afficher_details=False):
    A = np.array(A, dtype=float)
    b = np.squeeze(np.array(b, dtype=float)).reshape(-1, 1)
    matrice_aug = np.hstack([A, b])
    iterations = []
    
    for k in range(n-1,-1,-1):
        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
        matrice_aug[k, n] = matrice_aug[k, n]/pivot
        matrice_aug[k, k] = 1

        # Élimination au dessus de la diagonale
        for j in range(k-1,max(k-1-m,-1),-1):
            matrice_aug[j,n]=matrice_aug[j,n]-matrice_aug[j,k]*matrice_aug[k,n]
            matrice_aug[j,k]=0
    
        # Enregistrer les détails si demandé
        if afficher_details:
            iterations.append(matrice_aug.copy())


 # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations



# Gauss-Jordan pour matrices bande symétriques definies positives 
def gauss_jordan_bande_symétrique_definie_positive(A, b, n, m, afficher_details=False):
    # Conversion en float64 pour éviter les erreurs de calcul
    A = np.array(A, dtype=np.float64)
    b = np.squeeze(np.array(b, dtype=np.float64)).reshape(-1, 1)
    
    # Créer la matrice augmentée (A|b)
    matrice_aug = np.hstack([A, b])
    
    # Liste pour enregistrer les étapes intermédiaires si nécessaire
    iterations = []
    
    # Appliquer l'élimination de Gauss-Jordan sur la matrice augmentée
    for k in range(n):
        # Normaliser la ligne k (ligne pivot)
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
        
        # Diviser chaque élément de la ligne par le pivot pour normaliser
        matrice_aug[k, :] =(matrice_aug[k, :])/pivot
        
        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m+1, n)): 
            matrice_aug[i, :]=matrice_aug[i, :]-matrice_aug[k, :]*matrice_aug[i,k]
        
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
            iterations.append(matrice_aug.copy())
        
    # Élimination au dessus de la diagonale 
    for i in range(n-1,0,-1):
        for j in range(i-1,max(i-m-1,-1),-1):
            matrice_aug[j,n]=matrice_aug[j,n]-matrice_aug[j,i]*matrice_aug[i,n]
            matrice_aug[j,i]=0
    
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
                iterations.append(matrice_aug.copy())
    
    # La dernière colonne de la matrice augmentée est la solution
    x = matrice_aug[:, -1]
    return x, iterations









def gauss_jordan_dense_inverse(matrice, n, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])
    iterations = []  # Liste pour stocker les matrices à chaque itération
    
    for k in range(n):
        #pivotage partiel
        # Trouver le pivot maximal
        max_pivot_index = np.argmax(np.abs(matrice_aug[k:, k])) + k
        # Vérifier que que le pivot n'est pas null
        if matrice_aug[max_pivot_index, k] == 0:
            raise ValueError("La matrice est singulière ou mal conditionnée.")
        #échanger les lignes
        matrice_aug[[k, max_pivot_index], :] = matrice_aug[[max_pivot_index, k], :]

        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        matrice_aug[k, :] = matrice_aug[k, :]/pivot
        # Éliminer les autres lignes
        for i in range(n):
            if i != k:
                matrice_aug[i, :] = matrice_aug[i, :]-matrice_aug[i, k] * matrice_aug[k, :]
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:, n:]
    return inverse, iterations

#la fonction gauss_jordan_sans_pivotage est utlisée dans le cas d'une matrice diagonale dominante par ligne(ou|et) par colonne et dans le cas d'une matrice symétrique
def gauss_jordan_sans_pivotage_inverse(matrice, n, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    # Liste pour stocker les itérations si afficher_details est True
    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
            
            # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot

        # Élimination des autres éléments dans la colonne k
        for i in range(n):
            if i != k:
                matrice_aug[i, :] = matrice_aug[i, :] - matrice_aug[i, k] * matrice_aug[k, :]

        # Sauvegarder l'état actuel de la matrice augmentée si afficher_details est True
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:,n:]
    return inverse, iterations

def gauss_jordan_diagonale_inverse(matrice,n,afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    # Liste pour stocker les étapes
    iterations = []

    for k in range(n):
        # Normalisation de la ligne par le pivot
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
        
        matrice_aug[k, :] = matrice_aug[k, :] / pivot
        matrice_aug[k, k]=1
        # Enregistrer l'état actuel de la matrice augmentée si afficher_details est True
        if afficher_details:
            iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:,n:]
    return inverse, iterations

# Gauss-Jordan pour matrices triangulaires supérieures
def gauss_jordan_triangulaire_sup_inverse(matrice, n, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    if not resultat:
        for k in range(n-1,-1,-1):
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
            
            # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot

            for j in range(k-1,-1,-1):
                # aug_matrix[i-1,:]=aug_matrix[i-1,:]-aug_matrix[i,:]*aug_matrix[i-1,i]
                matrice_aug[j,:]=matrice_aug[j,:]-matrice_aug[j,k]*matrice_aug[k,:]
                matrice_aug[j,k]=0
            
            # Sauvegarder l'étape si nécessaire
            if afficher_details:
                iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:,n:]
    return inverse, iterations

# Gauss-Jordan pour matrices triangulaires inférieures
def gauss_jordan_triangulaire_inf_inverse(matrice, n, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot à l'étape {k} est nul, ce qui empêche de résoudre le système.")
                
                # Normaliser la ligne k
            matrice_aug[k, :] = matrice_aug[k, :] / pivot
            #if afficher_details:
            #    iterations.append(matrice_aug.copy())

        for i in range(k+1,n):
            # aug_matrix[i-1,:]=aug_matrix[i-1,:]-aug_matrix[i,:]*aug_matrix[i-1,i]
            matrice_aug[i,:]=matrice_aug[i,:]-matrice_aug[i,k]*matrice_aug[k,:]
            matrice_aug[i,k]=0

        # Sauvegarder l'étape si nécessaire
        if afficher_details:
                iterations.append(matrice_aug.copy())

     # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:, n:]
    return inverse, iterations

# Gauss-Jordan pour matrices bande
def gauss_jordan_bande_inverse(matrice, n, m, afficher_details=False): 
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    iterations = []

    for k in range(n):  # Parcours des lignes pour Gauss-Jordan
        #pivotage partiel
        # Trouver le pivot maximal
        max_pivot_index = np.argmax(np.abs(matrice_aug[k:, k])) + k
        # Vérifier que que le pivot n'est pas null
        if matrice_aug[max_pivot_index, k] == 0:
            raise ValueError("La matrice est singulière ou mal conditionnée.")
        #échanger les lignes
        matrice_aug[[k, max_pivot_index], :] = matrice_aug[[max_pivot_index, k], :]

        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        matrice_aug[k, :] = matrice_aug[k, :]/pivot

        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m + 1, n)): 
            matrice_aug[i, :]=matrice_aug[i, :]-matrice_aug[k, :]*matrice_aug[i,k]
        
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
            iterations.append(matrice_aug.copy())
    # Élimination au dessus de la diagonale 
    for i in range(n-1,0,-1):
        for j in range(i-1,-1,-1):
            matrice_aug[j,:]=matrice_aug[j,:]-matrice_aug[j,i]*matrice_aug[i,:]
            matrice_aug[j,i]=0
    
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
                iterations.append(matrice_aug.copy())
    # Extraction de la solution
    inverse= matrice_aug[:, n:]
    return inverse, iterations



# Gauss-Jordan pour matrices demi-bande inférieures 
def gauss_jordan_demi_bande_inf_inverse(matrice, n, m, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    iterations = []

    resultat = np.all(np.diag(matrice_aug) == 1)

    for k in range(n):  # Parcours des lignes pour Gauss-Jordan
        # Normaliser la ligne k
        if not resultat:
            pivot = matrice_aug[k, k]
            if pivot == 0:
                raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
            matrice_aug[k, :] = matrice_aug[k, :]/pivot

        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m + 1, n)): 
            matrice_aug[i, :]=matrice_aug[i, :]-matrice_aug[k, :]*matrice_aug[i,k]
            matrice_aug[i,k]=0


        # Enregistrer les détails si demandé
        if afficher_details:
            iterations.append(matrice_aug.copy())

    # La dernière colonne de la matrice augmentée est la solution
    inverse = matrice_aug[:, n:]
    return inverse, iterations

# Gauss-Jordan pour matrices demi-bande supérieures 
def gauss_jordan_demi_bande_sup_inverse(matrice, n, m, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    iterations = []
    
    for k in range(n-1,-1,-1):
        # Normaliser la ligne k
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
        matrice_aug[k, :] = matrice_aug[k, :]/pivot
        matrice_aug[k, k] = 1

        # Élimination au dessus de la diagonale
        for j in range(k-1,max(k-1-m,-1),-1):
            matrice_aug[j,:]=matrice_aug[j,:]-matrice_aug[j,k]*matrice_aug[k,:]
            matrice_aug[j,k]=0
    
        # Enregistrer les détails si demandé
        if afficher_details:
            iterations.append(matrice_aug.copy())


 # La dernière colonne de la matrice augmentée est la solution
    inverse = matrice_aug[:, n:]
    return inverse, iterations


# Gauss-Jordan pour matrices bande symétriques definies positives 
def gauss_jordan_bande_symétrique_definie_positive_inverse(matrice, n, m, afficher_details=False):
    identite = np.eye(n)
    matrice_aug = np.hstack([matrice,identite])

    # Liste pour enregistrer les étapes intermédiaires si nécessaire
    iterations = []
    
    # Appliquer l'élimination de Gauss-Jordan sur la matrice augmentée
    for k in range(n):
        # Normaliser la ligne k (ligne pivot)
        pivot = matrice_aug[k, k]
        if pivot == 0:
            raise ValueError(f"Le pivot est nul à l'étape {k}, impossible de continuer.")
        
        # Diviser chaque élément de la ligne par le pivot pour normaliser
        matrice_aug[k, :] =(matrice_aug[k, :])/pivot
        
        # Élimination au dessous de la diagonale 
        for i in range(k + 1, min(k + m+1, n)): 
            matrice_aug[i, :]=matrice_aug[i, :]-matrice_aug[k, :]*matrice_aug[i,k]
        
        # Si on veut afficher les étapes intermédiaires
        if afficher_details:
            iterations.append(matrice_aug.copy())
        
    # Élimination au dessus de la diagonale 
    for i in range(n-1,0,-1):
       for j in range(i-1,max(i-m-1,-1),-1):
        matrice_aug[j,:]=matrice_aug[j,:]-matrice_aug[j,i]*matrice_aug[i,:]
        matrice_aug[j,i]=0
    
        # Si on veut afficher les étapes intermédiaires
       if afficher_details:
                iterations.append(matrice_aug.copy())
    
    # La dernière colonne de la matrice augmentée est la solution
    inverse= matrice_aug[:, n:]
    return inverse, iterations













































"""
def gauss_jordan_inverse(matrice, n, afficher_details=False):
    aug_matrix = np.hstack([matrice, np.eye(n)])
    iterations = []  # Liste pour stocker les matrices à chaque itération
    
    for k in range(n):
        # Trouver l'indice de la ligne avec le pivot maximal
        max_pivot_index = np.argmax(np.abs(aug_matrix[k:, k])) + k
        
        # Échanger les lignes si le pivot est nul
        if aug_matrix[max_pivot_index, k] == 0:
            # Gestion du cas où toutes les entrées sous la colonne k sont nulles
            continue
        else:
            aug_matrix[[k, max_pivot_index], :] = aug_matrix[[max_pivot_index, k], :]
        
        # Échelonner la colonne k
        pivot = aug_matrix[k, k]

        # Diviser chaque élément de la ligne par le pivot
        for j in range(2 * n): 
            aug_matrix[k, j] /= pivot
        
        # Éliminer les autres colonnes en dessous et au-dessus de la diagonale
        for i in range(k):
            ratio = aug_matrix[i, k]
            for j in range(2 * n):
                aug_matrix[i, j] -= ratio * aug_matrix[k, j]
        for i in range(k+1, n):
            ratio = aug_matrix[i, k]
            for j in range(2 * n):
                aug_matrix[i, j] -= ratio * aug_matrix[k, j]
        
        # Afficher les détails si demandé
        if afficher_details:
            iterations.append(aug_matrix.copy())
    
    inverse_matrix = aug_matrix[:, n:]
    
    #if afficher_details:
    #    return inverse_matrix, iterations
    #else:
    return inverse_matrix, iterations



def multiplication_bande_inverse(A, n, m, afficher_details=False):
    inverse_A, iterations = gauss_jordan_inverse(A, n, afficher_details)  # Inclure afficher_details ici
    print("Matrice Inverse A :")
    print(inverse_A)
    
    resultat = np.zeros((n, n))
    
    for i in range(n):
        for k in range(max(0, i - m), min(n, i + m + 1)):
            resultat[i, i] += A[i, k] * inverse_A[k, i]
    
    if afficher_details:
        print("\nDétails des itérations de Gauss-Jordan :")
        for idx, mat in enumerate(iterations):
            print(f"Iteration {idx + 1} :\n{mat}\n")
    
    return resultat, inverse_A
"""