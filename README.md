# mlflow-tensorflow

## 実行コマンド

1. コードを直接実行

```bash
python train.py -s 80
```

2. MLflow Projectsを介して実行

```bash
mlflow run https://github.com/HighBridgeDragon/mlflow-tensorflow --experiment_id={} -v origin/main -P s=80
```
