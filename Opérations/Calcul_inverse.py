import streamlit as st
import numpy as np
import FLASK as f
import pandas as pd
from Fonctions import Calcul_résultat as cr
from Fonctions import Type_matrice as tm
from Fonctions import Fraction as FR



def app():
    st.header("Calcul de l'inverse: A\u207B\u00B9")
    selected_option = st.radio("Sélectionnez une méthode", ["Manuel", "Aléatoire","Importation fichier (.CSV)"])
    matrice_html = f.matriceA_html()
    

    if selected_option == "Manuel":
        st.components.v1.html(matrice_html, width=300, height=300, scrolling=True)
        afficher_details = st.checkbox("Afficher détails")

        if st.button("Résultat"):
            try:
                A = np.array(f.get_matrix_values_A(), dtype=float)
                n = A.shape[0]
                matrix_type = tm.determine_matrix_type(A)
                st.success(tm.message_info(matrix_type))

                if np.linalg.det(A) == 0:
                    st.warning("La matrice A n'est pas inversible")
                else:
                    if matrix_type[0] == "Bande":
                        inverse, iterations = cr.gauss_jordan_bande_inverse(A, n, matrix_type[1], afficher_details)
                        
                    elif matrix_type[0] == "Demi-bande inférieure":
                        inverse, iterations = cr.gauss_jordan_demi_bande_inf_inverse(A, n, matrix_type[1], afficher_details)

                    elif matrix_type[0] == "Demi-bande supérieure":
                        inverse, iterations = cr.gauss_jordan_demi_bande_sup_inverse(A, n, matrix_type[1], afficher_details)
                       
                    elif matrix_type[0] == "Bande symétrique définie positive":
                        inverse, iterations = cr.gauss_jordan_bande_symétrique_definie_positive_inverse(A, n, matrix_type[1], afficher_details)
                                
                    elif matrix_type == "Triangulaire inférieure":
                        inverse, iterations = cr.gauss_jordan_triangulaire_inf_inverse(A, n, afficher_details)
                       
                    elif matrix_type == "Triangulaire supérieure":
                        inverse, iterations = cr.gauss_jordan_triangulaire_sup_inverse(A, n, afficher_details)
                       
                    elif matrix_type == "Diagonale":
                        inverse, iterations = cr.gauss_jordan_diagonale_inverse(A, n, afficher_details)
                      
                    elif matrix_type == "Dense":
                        inverse, iterations = cr.gauss_jordan_dense_inverse(A, n, afficher_details)

                    elif matrix_type == "d'identité":
                        inverse = A

                    # pour une matrice : Symétrique , Symétrique définie positive , Diagonale dominante par ligne , Diagonale dominante par colonne et Diagonale dominante
                    else:
                        inverse, iterations = cr.gauss_jordan_sans_pivotage_inverse(A, n, afficher_details)
                                
                    

                    if afficher_details and matrix_type != "d'identité":
                                for idx, iteration in enumerate(iterations):
                                    st.write(f"Étape {idx + 1} :")
                                    st.dataframe(FR.matrix_to_fraction(iteration))
                    if inverse is not None:
                        st.subheader("Matrice Inverse (A⁻¹) :")
                        st.dataframe(FR.matrix_to_fraction(inverse))

                        # Générer le rapport en fonction du type détecté
                        if matrix_type[0] in ["Bande", "Bande symétrique définie positive"]:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type[0]} de largeur de bande m= {matrix_type[1]}
Taille de la matrice : {n}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and iterations:
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"
                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        elif matrix_type[0] in ["Demi-bande inférieure", "Demi-bande supérieure"]:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type[0]} de largeur de demi-bande m= {matrix_type[1]}
Taille de la matrice : {n}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and iterations:
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"

                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        else:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type}
Taille de la matrice : {n}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and matrix_type != "d'identité":
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"
                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        # Télécharger le fichier bilan
                        st.download_button(
                            label="Télécharger le rapport en .txt",
                            data=rapport.encode('utf-8'),
                            file_name="rapport_inverse.txt",
                            mime="text/plain"
                        )


            except ValueError as e:
                st.error(f"Une erreur est survenue lors de la conversion des données de la matrice A : {str(e)}")
            except Exception as ex:
                st.error(f"Une erreur imprévue est survenue lors du calcul de l'inverse de la matrice : {str(ex)}")



    elif selected_option == "Aléatoire":
        matrix_type = st.selectbox(
            "Sélectionnez le type de matrice",
            [
                "Dense", "Symétrique", "Symétrique définie positive", "Diagonale", "Diagonale dominante", 
                "Triangulaire supérieure", "Triangulaire inférieure",
                "Bande", "Bande symétrique définie positive", "Demi-bande inférieure", "Demi-bande supérieure" 
            ]
        )
        # Créer des colonnes pour aligner les champs
        col1, col2 = st.columns(2)

        # Taille de la matrice dans la première colonne
        with col1:
            size = st.number_input(
                "Entrez la taille de la matrice (n x n)", 
                min_value=2, max_value=10, step=1, value=3
            )

        # Largeur de la bande ou demi-bande dans la deuxième colonne
        with col2:
            if matrix_type in ["Bande", "Bande symétrique définie positive"]:
                bande_width = st.number_input(
                    "Largeur de la bande (m)", 
                    min_value=1, max_value=size - 1, step=1, value=1
                )
                demi_bande_width = None
            elif matrix_type in ["Demi-bande inférieure", "Demi-bande supérieure"]:
                demi_bande_width = st.number_input(
                    "Largeur de la demi-bande (m)", 
                    min_value=1, max_value=size - 1, step=1, value=1
                )
                bande_width = None
            else:
                bande_width = None
                demi_bande_width = None

        st.write("Définir les limites des valeurs de la matrice")
        # Entrées pour la valeur minimale et maximale
        col1, col2, a, aa = st.columns(4)
        with col1:
            min = st.number_input("Valeur minimale", min_value=1, max_value=99999, step=1, value=1)
        with col2:
            max = st.number_input("Valeur maximale", min_value=2, max_value=100000, step=1, value=10)

        # Vérification de la validité des valeurs
        if min > max:
            st.warning(f"La valeur minimale ne peut pas dépasser la valeur maximale. Les deux valeurs ont été automatiquement échangées.")
            min, max = max, min
        elif min == max:
            st.warning(f"La matrice sera remplie avec la même valeur pour tous ses éléments.")
        else:
            st.success(f"Les valeurs seront générées entre {min} et {max}.")

        
        afficher_details = st.checkbox("Afficher détails")

        if st.button("Générer et Calculer l'Inverse"):
            try:
                # Génération de la matrice spécifique
                A = tm.generate_specific_matrix(min, max, matrix_type, size, bande_width, demi_bande_width)

                st.subheader(f"Matrice A :")
                st.dataframe(FR.matrix_to_fraction(A))

                # Vérification du type
                detected_type = tm.determine_matrix_type(A)
                #st.info(f"Type détecté : {detected_type}")

                # Vérification de l'inversibilité
                if np.linalg.det(A) == 0:
                    st.warning("La matrice A n'est pas inversible.")
                else:
                    if matrix_type == "Bande":
                        inverse, iterations = cr.gauss_jordan_bande_inverse(A, size, bande_width, afficher_details)
                        
                    elif matrix_type == "Demi-bande inférieure":
                        inverse, iterations = cr.gauss_jordan_demi_bande_inf_inverse(A, size, demi_bande_width, afficher_details)

                    elif matrix_type == "Demi-bande supérieure":
                        inverse, iterations = cr.gauss_jordan_demi_bande_sup_inverse(A, size, demi_bande_width, afficher_details)
                       
                    elif matrix_type == "Bande symétrique définie positive":
                        inverse, iterations = cr.gauss_jordan_bande_symétrique_definie_positive_inverse(A, size, bande_width, afficher_details)
                                
                    elif matrix_type == "Triangulaire inférieure":
                        inverse, iterations = cr.gauss_jordan_triangulaire_inf_inverse(A, size, afficher_details)
                       
                    elif matrix_type == "Triangulaire supérieure":
                        inverse, iterations = cr.gauss_jordan_triangulaire_sup_inverse(A, size, afficher_details)
                       
                    elif matrix_type == "Diagonale":
                        inverse, iterations = cr.gauss_jordan_diagonale_inverse(A, size, afficher_details)
                      
                    elif matrix_type == "Dense":
                        inverse, iterations = cr.gauss_jordan_dense_inverse(A, size, afficher_details)

                    # pour une matrice : Symétrique , Symétrique définie positive , Diagonale dominante par ligne , Diagonale dominante par colonne et Diagonale dominante
                    else:
                        inverse, iterations = cr.gauss_jordan_sans_pivotage_inverse(A, size, afficher_details)

                    if afficher_details and iterations:
                                for idx, iteration in enumerate(iterations):
                                    st.write(f"Étape {idx + 1} :")
                                    st.dataframe(FR.matrix_to_fraction(iteration))
                    
                    if inverse is not None:
                        st.subheader("Matrice Inverse (A⁻¹) :")
                        st.dataframe(FR.matrix_to_fraction(inverse))

                        # Générer le rapport en fonction du type détecté
                        if matrix_type in ["Bande", "Bande symétrique définie positive"]:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type} de largeur de bande m= {bande_width}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and iterations:
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"

                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        elif matrix_type in ["Demi-bande inférieure", "Demi-bande supérieure"]:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type} de largeur de demi-bande m= {demi_bande_width}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and iterations:
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"

                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        else:
                            rapport = f"""
====================================================
                    RAPPORT
====================================================

Type de matrice : {matrix_type}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}
"""
                            if afficher_details and iterations:
                                rapport += "\n----------------------------------------------------\nDÉTAILS DES ÉTAPES\n"
                                for idx, iteration in enumerate(iterations):
                                    rapport += f"Étape {idx + 1} :\n"
                                    rapport += str(FR.matrix_to_fraction(iteration)) + "\n\n"

                            rapport += f"----------------------------------------------------\nMATRICE INVERSE (A⁻¹) :\n{FR.matrix_to_fraction(inverse)}\n\n===================================================="

                        # Télécharger le fichier bilan
                        st.download_button(
                            label="Télécharger le rapport en .txt",
                            data=rapport.encode('utf-8'),
                            file_name="rapport_inverse.txt",
                            mime="text/plain"
                        )


            except Exception as ex:
                st.error(f"Une erreur s'est produite lors de la génération de la matrice ou du calcul de son inverse : {str(ex)}")


    elif selected_option == "Importation fichier (.CSV)":
        st.subheader("Importer un fichier CSV pour calculer l'inverse d'une matrice")
        uploaded_file = st.file_uploader("Téléchargez votre fichier .CSV", type=["csv"])

        if uploaded_file:
            try:
                # Lire la première ligne pour les métadonnées
                first_line = uploaded_file.readline().decode('utf-8').strip()
                matrix_type, size = first_line.split(',')
                size = int(size)

                # Lire le reste du fichier
                df = pd.read_csv(uploaded_file, header=None)
                A = df.values  # Extraire la matrice


                # Vérifier les dimensions de la matrice
                if A.shape[0] != A.shape[1] or A.shape[0] != size:
                    st.error("Le nombre de lignes de la matrice A doit correspondre à la taille indiquée dans le fichier.")
                    return
                
                # Afficher la matrice si la taille est raisonnable
                if size <= 10:
                    st.subheader("Matrice A :")
                    st.dataframe(FR.matrix_to_fraction(A))

                # Déterminer le type de la matrice
                matrix_type = matrix_type.strip().replace('\ufeff', '')
                detected_type = tm.determine_matrix_type(A)
                st.success(tm.message_info(detected_type))

                if matrix_type in ["B", "BSDP"]: #and matrix_type_A in ["Bande", "Bande symétrique définie positive"]:
                    bande_width = detected_type[1]
                elif matrix_type in ["DBI", "DBS"]: #matrix_type in ["DBI", "DBS"] and matrix_type_A in ["Demi-bande inférieure", "Demi-bande supérieure"]
                    demi_bande_width = detected_type[1]
                else:
                    bande_width = None
                    demi_bande_width = None

                # Vérification de l'inversibilité
                if np.linalg.det(A) == 0:
                    st.warning("La matrice A n'est pas inversible.")
                else:
                    if matrix_type == "B":
                        inverse = cr.gauss_jordan_bande_inverse(A, size, bande_width)[0]
                        
                    elif matrix_type == "DBI":
                        inverse = cr.gauss_jordan_demi_bande_inf_inverse(A, size, demi_bande_width)[0]

                    elif matrix_type == "DBS":
                        inverse = cr.gauss_jordan_demi_bande_sup_inverse(A, size, demi_bande_width)[0]
                       
                    elif matrix_type == "BSDP":
                        inverse = cr.gauss_jordan_bande_symétrique_definie_positive_inverse(A, size, bande_width)[0]
                                
                    elif matrix_type == "TI":
                        inverse = cr.gauss_jordan_triangulaire_inf_inverse(A, size)[0]
                       
                    elif matrix_type == "TS":
                        inverse = cr.gauss_jordan_triangulaire_sup_inverse(A, size)[0]
                       
                    elif matrix_type == "DI":
                        inverse = cr.gauss_jordan_diagonale_inverse(A, size)[0]
                      
                    elif matrix_type == "DE":
                        inverse = cr.gauss_jordan_dense_inverse(A, size)[0]

                    elif matrix_type in ["S", "SDP", "DDL", "DDC", "DD"]:
                        inverse = cr.gauss_jordan_sans_pivotage_inverse(A, size)[0]
                    else:
                        st.error("Le type de matrice spécifié n'est pas pris en charge ou est mal défini")
                        return


                    st.subheader("Matrice inverse A\u207B\u00B9 :")
                    st.dataframe(FR.matrix_to_fraction(inverse))



                    if detected_type[0] in ["Bande", "Bande symétrique définie positive"]:
                        rapport = f"""
====================================================
                     RAPPORT
====================================================

Type de matrice : {detected_type[0]} de largeur de bande m= {bande_width}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}

----------------------------------------------------
MATRICE INVERSE (A\u207B\u00B9) :
{FR.matrix_to_fraction(inverse)}

====================================================
"""
                    elif detected_type[0] in ["Demi-bande inférieure", "Demi-bande supérieure"]:
                        rapport = f"""
====================================================
                     RAPPORT
====================================================

Type de matrice : {detected_type[0]} de largeur de demi-bande m= {demi_bande_width}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}

----------------------------------------------------
MATRICE INVERSE (A\u207B\u00B9) :
{FR.matrix_to_fraction(inverse)}

====================================================
"""
                    else:
                        rapport = f"""
====================================================
                     RAPPORT
====================================================

Type de matrice : {detected_type}
Taille de la matrice : {size}

----------------------------------------------------
MATRICE A :
{str(FR.matrix_to_fraction(A))}

----------------------------------------------------
MATRICE INVERSE (A\u207B\u00B9) :
{FR.matrix_to_fraction(inverse)}

====================================================
"""

                # Télécharger le rapport en .txt
                st.download_button(
                    label="Télécharger le rapport en .txt",
                    data=rapport.encode('utf-8'),
                    file_name="rapport_matrice_inverse.txt",
                    mime="text/plain"
                )
            except Exception as ex:
                st.error(f"Une erreur s'est produite lors du traitement du fichier : {str(ex)}")


def main():
    app()


if __name__ == "__main__":
    main()
