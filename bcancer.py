from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger le jeu de données sur le cancer du sein
data = load_breast_cancer()

# Créer un DataFrame pour les fonctionnalités (features)
X = pd.DataFrame(data.data, columns=data.feature_names)

# Créer une Série pour les étiquettes (labels)
y = pd.Series(data.target)

# Diviser le jeu de données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(x_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(x_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Afficher les résultats
print(f'Précision du modèle : {accuracy}')
print('Matrice de confusion :\n', conf_matrix)
print('Rapport de classification :\n', classification_rep)
