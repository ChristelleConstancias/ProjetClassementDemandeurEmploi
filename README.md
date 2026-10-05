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
`artifacts/pipeline_meilleur_recall_f1.joblib`, au format dictionnaire contenant les cles `pipeline` et `retraining_config`.

Lancer ensuite l'API depuis la racine du projet :

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Ouvrir `http://127.0.0.1:8000/docs` pour tester `POST /predict`. `GET /health` doit retourner 200. Si le chargement echoue, verifier que `artifacts/pipeline_meilleur_recall_f1.joblib` existe et que ses dependances sont installees.

### Tests et lint locaux

```powershell
python -m pip install -r requirements.txt pytest ruff
python -m compileall -q src tests
python -m ruff check --select E4,E7,E9,F src/api.py src/app.py src/data.py src/models.py src/train_model.py tests/
python -m pytest -q tests/
```

### CI/CD GitHub Actions

Les pull requests executent compilation, lint, tests et build de l'image sans publication. Un push sur `main` publie l'image dans GHCR, puis le job `deploy` la deploie. Ce dernier necessite un runner Linux auto-heberge portant les labels `self-hosted`, `linux` et `production`, avec Docker Compose et acces a GHCR.

Avant le premier deploiement, preparer sur ce runner :

- `/srv/retour-emploi/artifacts/pipeline_meilleur_recall_f1.joblib` (artefact valide, lisible par l'UID 10001 du conteneur) ;
- `/srv/retour-emploi/data/dataset_trajectoire_emploi_Sujet Examen CISIA.csv` (lisible par l'UID 10001) ;
- les droits d'ecriture pour l'UID 10001 sur `/srv/retour-emploi/outputs`.

Le fichier `docker-compose.yml` monte ces repertoires dans le conteneur et expose l'API uniquement sur `127.0.0.1:8000`; un proxy autorise doit fournir l'acces reseau requis. Les donnees et le modele ne sont pas integres a l'image. Le deploiement echoue volontairement si l'artefact ou le CSV manque.

Compose lance egalement le serveur de tracking MLflow sur `127.0.0.1:5000`. L'API le contacte sur le reseau interne Compose via `MLFLOW_TRACKING_URI=http://mlflow:5000`; `/retrain` journalise les parametres, metriques et le pipeline candidat avant d'activer le modele. Deux volumes nommes conservent la base SQLite MLflow et les artefacts apres le redemarrage des conteneurs. Le notebook utilise toujours le tracking SQLite local lorsqu'il est execute hors de Compose.
