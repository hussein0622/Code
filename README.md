# Système de Prédiction d'Insuffisance Cardiaque

Une application web professionnelle pour les médecins permettant de prédire le risque d'insuffisance cardiaque chez les patients.

## Caractéristiques

- Interface médicale professionnelle avec design moderne
- Système d'authentification sécurisé
- Validation complète des données d'entrée avec plages de valeurs médicales
- Visualisation des résultats avec graphiques
- Prédictions en temps réel
- Design responsive pour tous les appareils
- Traitement sécurisé des formulaires
- Journalisation des événements pour le suivi

## Instructions d'installation

1. Créer un environnement virtuel (recommandé):
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Exécuter l'application:
```bash
python app.py
```

4. Ouvrir votre navigateur et accéder à `http://localhost:5000`

## Informations de connexion

- Email: medecin@example.com
- Mot de passe: password123

## Paramètres du modèle

L'application utilise un modèle de forêt aléatoire (Random Forest) pour prédire le risque d'insuffisance cardiaque basé sur les paramètres suivants:

| Paramètre | Description | Plage de valeurs |
|-----------|-------------|-----------------|
| Age | Âge du patient | 18-100 ans |
| Anaemia | Présence d'anémie | Oui/Non |
| Creatinine Phosphokinase | Niveau de CPK dans le sang | 10-10000 U/L |
| Diabetes | Présence de diabète | Oui/Non |
| Ejection Fraction | Pourcentage de sang quittant le cœur à chaque contraction | 10-80% |
| High Blood Pressure | Présence d'hypertension | Oui/Non |
| Platelets | Nombre de plaquettes dans le sang | 50000-500000 per µL |
| Serum Creatinine | Niveau de créatinine sérique | 0.1-10 mg/dL |
| Serum Sodium | Niveau de sodium sérique | 110-150 mmol/L |
| Sex | Sexe du patient | Homme/Femme |
| Smoking | Statut tabagique | Oui/Non |
| Time | Période de suivi | 0-365 jours |

## Note

Cette application est un modèle fonctionnel. Pour une utilisation en production:
1. Remplacez le modèle factice par votre modèle entraîné
2. Configurez une clé secrète appropriée pour la production
3. Ajoutez des mesures de sécurité supplémentaires
4. Implémentez une base de données pour la gestion des utilisateurs
