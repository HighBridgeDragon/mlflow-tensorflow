# mlflow-tensorflow

## 実行コマンド

1. コードを直接実行

```bash
python train.py -s 80
```

2. MLflow Projectsを介して実行

```bash
mlflow run https://github.com/HighBridgeDragon/mlflow-tensorflow --experiment-id={} -v origin/main -P s=80
```

  conda と mlflowにパスが通った環境でのみ動作します。
  condaのインストールは、[miniconda](https://docs.conda.io/en/latest/miniconda.html)がおすすめ。
  
  mlflow は、以下のコマンドでインストールします。
```bash
  conda install mlflow
```

