# SUJET E6 : PINNs pour Pricing d'Options

**Difficulté :** 4/5 | **Domaine :** Probabilités, Machine Learning

**Auteur :** SUSINI | **Groupe :** 03-BOIVIN | **Promotion :** ING4 — ECE Paris, 2026

---

## Description

Ce projet implémente des **Physics-Informed Neural Networks (PINNs)** pour résoudre l'équation aux dérivées partielles (EDP) de **Black-Scholes** et ses extensions (Heston, Merton jump-diffusion) appliquées au pricing d'options financières.

Les PINNs intègrent l'EDP financière directement comme contrainte dans la fonction de loss, offrant une solution **mesh-free** pour le pricing de dérivés complexes.

**Résultats de référence :**
- 12.5% d'amélioration sur calls NASDAQ vs méthodes traditionnelles
- 59% d'amélioration sur puts américaines vs méthodes traditionnelles

---

## Objectifs

| Niveau | Objectif |
|--------|----------|
| **Minimum** | PINN en PyTorch pour résoudre l'EDP Black-Scholes (call européen), comparaison avec solution analytique |
| **Bon** | Extension aux puts américaines (problème à frontière libre), visualisation de la surface prix(spot, maturité) |
| **Excellent** | Extension au modèle de Heston ou Merton, pricing d'options exotiques, analyse de convergence |

---

## Structure du projet

```
PINNs_pour_Pricing_d_option/
│
├── README.md                          # Ce fichier
│
├── data/                              # Données de marché (optionnel)
│   └── nasdaq_options.csv             # Données options NASDAQ pour validation
│
├── notebooks/                         # Jupyter Notebooks (exploration & présentation)
│   ├── 01_black_scholes_baseline.ipynb    # EDP Black-Scholes + solution analytique
│   ├── 02_pinn_european_call.ipynb        # PINN pour call européen (objectif minimum)
│   ├── 03_pinn_american_put.ipynb         # PINN pour put américaine (frontière libre)
│   ├── 04_pinn_heston.ipynb               # Extension modèle de Heston (objectif excellent)
│   ├── 05_pinn_merton.ipynb               # Extension Merton jump-diffusion
│   └── 06_convergence_analysis.ipynb      # Analyse de convergence et comparaisons
│
├── src/                               # Code source Python
│   ├── __init__.py
│   │
│   ├── models/                        # Architectures des réseaux de neurones
│   │   ├── __init__.py
│   │   ├── pinn_base.py               # Classe de base PINN
│   │   ├── pinn_black_scholes.py      # PINN spécialisé Black-Scholes
│   │   ├── pinn_american.py           # PINN pour options américaines
│   │   └── pinn_heston.py             # PINN modèle de Heston
│   │
│   ├── equations/                     # Définition des EDPs financières
│   │   ├── __init__.py
│   │   ├── black_scholes.py           # EDP Black-Scholes + conditions aux limites
│   │   ├── american_option.py         # Contrainte de complémentarité (frontière libre)
│   │   ├── heston.py                  # EDP de Heston (volatilité stochastique)
│   │   └── merton.py                  # EDP de Merton (sauts de diffusion)
│   │
│   ├── analytical/                    # Solutions analytiques de référence
│   │   ├── __init__.py
│   │   ├── black_scholes_formula.py   # Formule de Black-Scholes (call/put européen)
│   │   └── binomial_tree.py           # Arbre binomial (référence put américaine)
│   │
│   ├── training/                      # Boucles d'entraînement et losses
│   │   ├── __init__.py
│   │   ├── loss_functions.py          # Loss PDE + boundary + initial conditions
│   │   ├── trainer.py                 # Boucle d'entraînement générique
│   │   └── callbacks.py               # Early stopping, logging, checkpoints
│   │
│   └── visualization/                 # Outils de visualisation
│       ├── __init__.py
│       ├── price_surface.py           # Surface prix(spot, maturité) 3D
│       └── convergence_plots.py       # Courbes de convergence de loss
│
├── tests/                             # Tests unitaires
│   ├── test_black_scholes_formula.py  # Validation solution analytique
│   ├── test_pinn_bs.py                # Tests PINN Black-Scholes
│   └── test_loss_functions.py         # Tests des fonctions de loss
│
├── results/                           # Résultats sauvegardés
│   ├── figures/                       # Graphiques générés
│   └── models/                        # Poids des modèles entraînés (.pt)
│
├── requirements.txt                   # Dépendances Python
└── report/
    └── rapport_E6_SUSINI.pdf          # Rapport final
```

---

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd PINNs_pour_Pricing_d_option

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
torch>=2.0.0
deepxde>=1.10.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
jupyter>=1.0.0
pandas>=2.0.0
```

---

## Utilisation

### Lancer les notebooks dans l'ordre

```bash
jupyter notebook notebooks/
```

1. **`01_black_scholes_baseline.ipynb`** — Comprendre l'EDP et la solution analytique
2. **`02_pinn_european_call.ipynb`** — Premier PINN, comparaison avec Black-Scholes
3. **`03_pinn_american_put.ipynb`** — Puts américaines avec frontière libre
4. **`04_pinn_heston.ipynb`** — Modèle de Heston (volatilité stochastique)
5. **`06_convergence_analysis.ipynb`** — Analyse comparative finale

### Entraînement rapide via script

```bash
python -m src.training.trainer --model black_scholes --epochs 10000
```

---

## Approche technique

### Architecture PINN

Un PINN est un réseau de neurones dont la loss combine :

```
L_total = λ_pde · L_pde + λ_bc · L_bc + λ_ic · L_ic + λ_data · L_data
```

- **`L_pde`** — résidu de l'EDP (Black-Scholes, Heston...)
- **`L_bc`** — conditions aux limites (payoff à maturité, barrières)
- **`L_ic`** — conditions initiales
- **`L_data`** — données de marché observées (optionnel)

### EDP Black-Scholes

```
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

avec la condition terminale : `V(S, T) = max(S - K, 0)` pour un call européen.

---

## Références

| Ressource | Description |
|-----------|-------------|
| [DeepXDE](https://deepxde.readthedocs.io/) | Bibliothèque PINNs Python (tutoriels inclus) |
| [DeepXDE Examples](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward.html) | Galerie d'exemples PDE forward/inverse |
| [PINN Option Pricing (arXiv)](https://arxiv.org/abs/2105.01937) | Paper de référence |
| [MathWorks PINN Tutorial (2025)](https://www.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html) | Tutoriel détaillé avec code |
| [PyTorch PINN Tutorial](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a) | Introduction pédagogique |
| Infer-101 | Black-Scholes et probabilités (référence théorique du cours) |
