# HMM_js de DEROUET Axel et MASSONNAT Maxime

# Veuillez trouver ci-joint notre rendu par rapport au projet js utilisant les modèles cachés de Markov afin de déterminer la probabilité, sur une phrase aléatoire pour un caractère donné, du caractère suivant pour un caractère de la phrase donnée.


# Le fichier TP_HMM_predictionTexte.js est le fichier éxécutant l'entrainement du modèle HMM qui, à partir texte mis en entrée, permet le choix d'un caractère dans la phrase et renvoie les trois probabilités les plus hautes concernant le caractère suivant. Il utilise la librairie Ramda et des pipes en utilisant une approche fonctionnelle.
# Le fichier permettant d'importer les textes aléatoires est le mots_filtrés.txt
# Enfin, hmm_model.json est le fichier qui permet d'améliorer l'entrainement en se basant sur les probabilités précédents l'itération, ainsi, il se crée après la première itération du fichier TP_HMM_PredictionTexte.js
