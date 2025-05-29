import streamlit as st

def main():
    st.title("Aide")

    # Section Résolution de Systèmes Linéaires
    st.markdown("""
    <h2 id="resolution-systemes">Résolution de Systèmes Linéaires</h2>
    <h3>1. Manuel</h3>
        <ul>
            <li>Entrez les valeurs de la matrice A et du vecteur b.</li>
            <li>Le système détecte automatiquement le type de la matrice entrée (Symétrique, Bande, etc.) et applique l'algorithme spécifique à ce type.</li>
            <li>Cochez "Afficher détails" pour voir la matrice à chaque itération de la résolution par Gauss-Jordan.</li>
        </ul>

    <h3>2. Aléatoire</h3>
        <ul>
            <li>Sélectionnez un type de matrice (Symétrique, Bande, Demi-bande inférieure, Demi-bande supérieure, etc.).</li>
            <li>Entrez la taille de la matrice, la "Valeur minimale" et la "Valeur maximale" des éléments. (Remarque : si "Valeur minimale" est supérieure à "Valeur maximale", les valeurs seront automatiquement permutées.)</li>
            <li>Pour les types "Bande", "Demi-bande inférieure" ou "Demi-bande supérieure", saisissez également la valeur de la bande m.</li>
            <li>Cochez "Afficher détails" pour voir la matrice à chaque itération.</li>
        </ul>

    <h3>3. Importation fichier (.CSV)</h3>
        <ul>
            <li>Préparez un fichier CSV contenant la matrice augmentée (le vecteur b concaténé avec la matrice A) avec la première ligne sous la forme suivante : "abréviation du type de la matrice, taille de la matrice".</li>
            <li id="abreviations">Les abréviations sont les suivantes : 
                <ul>
                    <li>DE : Dense</li>
                    <li>S : Dense Symétrique</li>
                    <li>SDP : Dense Symétrique Définie Positive</li>
                    <li>TI : Triangulaire Inférieur</li>
                    <li>TS : Triangulaire Supérieur</li>
                    <li>DI : Diagonale</li>
                    <li>DDL : Diagonale Dominante par Ligne</li>
                    <li>DDC : Diagonale Dominante par Colonne</li>
                    <li>B : Bande</li>
                    <li>BS : Bande Symétrique</li>
                    <li>BSDP : Bande Symétrique Définie Positive</li>
                    <li>DBI : Demi-Bande Inférieur</li>
                    <li>DBS : Demi-Bande Supérieur</li>
                </ul>
            </li>
            <li>Importez le fichier CSV et, si la taille de la matrice est inférieure ou égale à 10, la matrice augmentée, la matrice A et le vecteur b seront affichés, ainsi que le type de matrice détecté et le vecteur solution x.</li>
            <li>Si la taille de la matrice est supérieure à 10, seul le type de matrice et le vecteur solution x seront affichés.</li>
        </ul>
    """, unsafe_allow_html=True)
    st.video("videos/vid_resolution.mp4")
    # Section Calcul de l'inverse
    st.markdown("""
    <h2 id="calcul-inverse">Calcul de l'Inverse d'une Matrice</h2>
    <h3>1. Manuel</h3>
        <ul>
            <li>Entrez les valeurs de la matrice A.</li>
            <li>Le système détecte automatiquement le type de la matrice entrée (Symétrique, Bande, etc.) et applique l'algorithme spécifique à ce type.</li>
            <li>Cochez "Afficher détails" pour voir la matrice à chaque itération du calcul de l'inverse.</li>
        </ul>
    <h3>2. Aléatoire</h3>
        <ul>
            <li>Sélectionnez un type de matrice (Symétrique, Bande, Demi-bande inférieure, Demi-bande supérieure, etc.).</li>
            <li>Entrez la taille de la matrice, la "Valeur minimale" et la "Valeur maximale" des éléments. (Remarque : si "Valeur minimale" est supérieure à "Valeur maximale", les valeurs seront automatiquement permutées.)</li>
            <li>Pour les types "Bande", "Demi-bande inférieure" ou "Demi-bande supérieure", saisissez également la valeur de la bande m.</li>
            <li>Cochez "Afficher détails" pour voir la matrice à chaque itération.</li>
        </ul>

    <h3>3. Importation fichier (.CSV)</h3>
        <ul>
            <li>Préparez un fichier CSV contenant uniquement la matrice A, avec la première ligne sous la forme suivante : "<a href="#abreviations">abréviation du type de la matrice</a>, taille de la matrice".</li>
            <li>Importez le fichier CSV et, si la taille de la matrice est inférieure ou égale à 10, la matrice A sera affichée ainsi que l'inverse calculé.</li>
            <li>Si la taille de la matrice est supérieure à 10, seul le type de matrice et l'inverse de la matrice seront affichés.</li>
        </ul>
    """, unsafe_allow_html=True)
    st.video("videos/vid_inverse.mp4")
    # Section Explication des Erreurs
    st.markdown("""
    <h2>Explication des Erreurs</h2>
    <ul>
        <li><b>Le nombre de lignes de la matrice A doit correspondre à la taille du vecteur b</b> : La condition vérifie si le nombre de lignes de A correspond à la taille de b. Si ce n’est pas le cas, il est impossible de résoudre le système linéaire.</li>
        <li><b>Le système possède une infinité de solutions</b> : La matrice A est singulière avec un vecteur b nul ou son rang est inférieur au nombre de ses colonnes.</li>
        <li><b>Le système n'a pas de solution</b> : Si le déterminant de A est nul, mais que la norme de b n’est pas nulle, alors le système est incompatible (aucune solution possible).</li>
        <li><b>La valeur minimale ne peut pas dépasser la valeur maximale</b> : Cela signifie que l'utilisateur a saisi une valeur minimale plus grande que la valeur maximale, ce qui est incohérent. Pour corriger automatiquement cette erreur, les deux valeurs ont été permutées afin de garantir la logique des données.</li>
        <li><b>La matrice sera remplie avec la même valeur pour tous ses éléments</b> : Cette situation survient lorsque l'utilisateur définit les mêmes valeurs pour le minimum et le maximum. Cela entraînera la génération d'une matrice où tous les éléments sont identiques, ce qui pourrait affecter les résultats du calcul.</li>
        <li><b>Le type de matrice spécifié n'est pas pris en charge ou est mal défini</b> : Ce message indique que le type de matrice sélectionné par l'utilisateur n'est pas reconnu ou n'est pas pris en charge par le programme. L'utilisateur doit vérifier et corriger la définition du type de matrice dans les paramètres fournis.</li>
        <li><b>Une erreur est survenue lors de la conversion des données de la matrice A ou du vecteur b</b> : Cette erreur se produit lorsqu'il y a une tentative de conversion des données de la matrice A ou du vecteur b en type float, mais les données saisies ne sont pas valides pour cette conversion. Il est essentiel que l'utilisateur entre des valeurs numériques valides.</li>
        <li><b>Une erreur imprévue est survenue lors de la résolution du système</b> : Ce message indique qu'une exception imprévue a interrompu le processus de résolution du système linéaire. Cela peut être dû à des problèmes dans les données d'entrée ou à un dysfonctionnement interne. L'utilisateur est invité à vérifier ses données pour s'assurer qu'elles sont correctes.</li>
        <li><b>Une erreur s'est produite lors de la génération de la matrice ou de la résolution du système</b> : Cette erreur peut survenir pendant la génération de la matrice ou la résolution du système linéaire. Elle peut être causée par des paramètres invalides ou des problèmes internes. L'utilisateur est encouragé à vérifier les valeurs fournies et à tenter une nouvelle exécution.</li>
        <li><b>Le nombre de lignes de la matrice A doit correspondre à la taille du vecteur b ainsi qu'à la taille indiquée dans le fichier</b> : Cette erreur survient si la matrice A a un nombre de lignes qui ne correspond pas à la taille du vecteur b ou à la taille spécifiée dans le fichier. La cohérence des dimensions est essentielle pour que le système linéaire soit valide.</li>
        <li><b>Une erreur s'est produite lors du traitement du fichier</b> : Cette erreur se produit lorsqu'un problème survient pendant la lecture ou le traitement du fichier CSV, ce qui peut être dû à un format incorrect, des données manquantes ou mal formées. L'utilisateur doit vérifier le fichier pour s'assurer qu'il respecte les spécifications attendues.</li>
        <li><b>La matrice A n'est pas inversible</b> : Cette erreur indique que la matrice A ne peut pas être inversée, car son déterminant est égal à zéro. Cela signifie que la matrice est singulière et qu'il n'existe pas d'inverse pour celle-ci.</li>
        <li><b>Une erreur imprévue est survenue lors du calcul de l'inverse de la matrice</b> : Cette erreur se produit si un problème survient pendant le calcul de l'inverse de la matrice, particulièrement pour les matrices de type bande. Cela peut être dû à une matrice non inversible ou à un problème avec la méthode de calcul choisie.</li>
        <li><b>Une erreur s'est produite lors de la génération de la matrice ou du calcul de son inverse</b> : Cette erreur se déclenche lorsque la matrice générée ou son inverse ne peut pas être calculé correctement, souvent en raison de paramètres invalides (telles que des dimensions incorrectes ou une matrice non inversible) ou d'une erreur dans le processus de génération.</li>
    </ul>

    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()