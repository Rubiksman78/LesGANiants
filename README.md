# LesGANiants
Compétition CeterisParibus Face Challenge visant à modifier les attributs de visages tirés de FFHQ pour un dataset plus équitable.

# Framework utilisée
- Inversion du générateur de StyleGan2 avec un encodeur et une optimisation directe du code latent de chacune des 70 images du dataset.
- Recherche des directions et des plans de séparation dans l'espace latent à l'aide de la méthode proposée par InterFaceGAN
