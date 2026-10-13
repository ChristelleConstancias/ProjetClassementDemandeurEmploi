# Retour a l'emploi : architecture cible et prototype interne

Ce depot contient un prototype d'inference du scenario multimodal 1. La prediction est une **aide a la decision** : aucune decision administrative automatique ne
doit etre prise sans revue par un conseiller. Le choix du modele et la
promotion d'une nouvelle version sont des actions humaines documentees.

## Flux et responsabilites

Le guichet (ou son backend autorise) lit les donnees du referentiel national :
l'API d'IA ne recoit que les champs necessaires au calcul, ne se connecte pas directement en ecriture à la base transactionnelle et ne stocke pas d'identifiant national. L'identifiant de session permet de retrouver les predictions d'un entretien. La correction est un **resultat confirme**, pas une prediction substituee automatiquement à la verite terrain. 

## Execution locale (prototype)

### API seule, sans Docker

Depuis la racine du projet, installer les dependances avec
`pip install -r requirements.txt`. Executer les cellules du notebook jusqu'a l'export du pipeline : l'API attend
`artifacts/pipeline_prediction_retour_emploi.joblib`, au format dictionnaire contenant les cles `pipeline` et `retraining_config`.

Lancer ensuite l'API depuis la racine du projet :

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Ouvrir `http://127.0.0.1:8000/docs` pour tester `POST /predict`. `GET /health` doit retourner 200. Si le chargement echoue, verifier que `artifacts/pipeline_prediction_retour_emploi.joblib` existe et que ses dependances sont installees.

### Interface Streamlit

Depuis la racine du projet, lancer :

```powershell
python -m streamlit run src/app.py
```

La navigation contient trois pages : `Nouvelle prédiction`, `Historique` (filtres et feedback) et `Gestion du modèle` (état et réentraînement).

### Tests et lint locaux

```powershell
python -m pip install -r requirements.txt pytest ruff
python -m compileall -q src tests
python -m ruff check --select E4,E7,E9,F src/api.py src/app.py src/data.py src/models.py tests/
python -m pytest -q tests/
```

### CI/CD GitHub Actions

Les pull requests executent compilation, lint, tests et build de l'image sans publication. Un push sur `main` publie les tags `<commit-sha>` et `latest` dans GHCR, puis appelle le Deploy Hook Render. Ajouter l'URL du hook dans les secrets GitHub sous le nom `RENDER_DEPLOY_HOOK_URL` et desactiver l'auto-deploiement Render pour garder cet ordre.

Sur Render, creer un Web Service de type **Existing Image**, configure sur `ghcr.io/<proprietaire>/<depot>:latest`, avec un health check sur `/health`. Le package GHCR doit etre public ou Render doit avoir des identifiants de lecture du registre. Ajouter un disque persistant monte sur `/var/data` et configurer ces variables d'environnement :

- `MODEL_PATH=/var/data/artifacts/pipeline_prediction_retour_emploi.joblib`
- `DATA_DIR=/var/data/data`
- `OUTPUTS_DIR=/var/data/outputs`
- `MLFLOW_DB_FILE=/var/data/mlflow.db`
- `MLFLOW_TRACKING_URI=sqlite:////var/data/mlflow.db`
- `USAGER_ID_HMAC_KEY` avec une cle secrete propre a l'environnement de production.

Les dossiers `artifacts/` et `data/` sont exclus de Git et ne sont pas integres a l'image. Avant d'utiliser l'API, provisionner le disque Render avec l'artefact modele et les fichiers de donnees necessaires, dans les chemins ci-dessus. Ne pas versionner ces fichiers, qui peuvent contenir des donnees sensibles. Le disque doit aussi permettre l'ecriture dans `/var/data/outputs`. Le Dockerfile ecoute sur la variable `PORT` fournie par Render.

Le `docker-compose.yml` reste disponible pour le deploiement local ou sur un serveur gere : il lance l'API et un serveur MLflow distinct, contrairement au deploiement Render qui utilise le tracking SQLite sur disque persistant.
