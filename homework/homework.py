# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

# limpiar_datasets.py
import pandas as pd
from pathlib import Path

def limpiar_dataset_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Renombrar objetivo
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})

    # Eliminar columna ID si existe
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Asegurar EDUCATION numérica y agrupar >4 -> 4 (others)
    if 'EDUCATION' in df.columns:
        # convertir a numérico si viene como string
        df['EDUCATION'] = pd.to_numeric(df['EDUCATION'], errors='coerce')
        df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4

    # Eliminar registros con información no disponible
    df = df.dropna().reset_index(drop=True)
    return df


def cargar_y_procesar(train_csv="files/input/train_data.csv",
                      test_csv="files/input/test_data.csv",
                      ):
    """Carga y limpia train/test; opcionalmente guarda los CSV limpios."""
    df_train = limpiar_dataset_csv(train_csv)
    df_test = limpiar_dataset_csv(test_csv)


    return df_train, df_test

# Dividir datasets en X/y (train y test)
from typing import Tuple, Optional

def dividir_en_xy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "default"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
  
    if target_col not in df_train.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en df_train. "
                         f"Columnas disponibles: {list(df_train.columns)}")

    # Separar X e y en train
    X_train = df_train.drop(columns=[target_col]).copy()
    y_train = pd.to_numeric(df_train[target_col], errors="coerce").astype("Int64")

    # Separar X e y en test (si existe la columna)
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col]).copy()
        y_test = pd.to_numeric(df_test[target_col], errors="coerce").astype("Int64")
    else:
        X_test = df_test.copy()
        y_test = None

    # Alinear columnas de X_train y X_test (mismo set y mismo orden)
    common_cols = X_train.columns.intersection(X_test.columns)
    if len(common_cols) == 0:
        raise ValueError("X_train y X_test no comparten columnas. Revisa el preprocesamiento.")

    # Avisar si hay columnas faltantes/extras y alinear
    faltantes_en_test = [c for c in X_train.columns if c not in X_test.columns]
    extras_en_test = [c for c in X_test.columns if c not in X_train.columns]
    if faltantes_en_test or extras_en_test:
        print("[Aviso] Alineando columnas:")
        if faltantes_en_test:
            print(f" - Columnas presentes en train y faltantes en test: {faltantes_en_test}")
        if extras_en_test:
            print(f" - Columnas presentes en test y no en train: {extras_en_test}")

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()
    
    return X_train, y_train, X_test, y_test 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC


def crear_pipeline_rf(categorical_cols, numeric_cols) -> Pipeline:
    """
    Crea un pipeline con:
    - OneHotEncoder para categóricas (con imputación)
    - SimpleImputer para numéricas
    - RandomForestClassifier
    """
    
    cat_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_preprocess, categorical_cols),
        ("num", num_preprocess, numeric_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),  # estandariza todo lo que sale del preprocessor
        ("pca", PCA()),                # usa todas las componentes
        ("select", SelectKBest(score_func=f_classif, k=23)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    return model


import os
import gzip
import pickle
import numpy as np
def find_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No encontré el modelo. Probé: " + ", ".join(MODEL_PATHS)
    )

def load_model(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

POS_LABEL = None
def infer_setting(y):
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes <= 2:
        if POS_LABEL is not None:
            pos_label = POS_LABEL
        else:
            try:
                numeric = np.array(classes, dtype=float)
                if set(numeric.tolist()) == {0.0, 1.0}:
                    pos_label = 1
                else:
                    pos_label = classes[-1]
            except Exception:
                pos_label = classes[-1]
        return {"average": "binary", "pos_label": pos_label}
    
    else:
        return {"average": "weighted", "pos_label": None}

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(model, X, y, dataset_name):
    y_pred = model.predict(X)

    # balanced accuracy (soporta binario y multiclase)
    bacc = balanced_accuracy_score(y, y_pred)

    # Configuración para precision/recall/f1
    cfg = infer_setting(y)
    average = cfg["average"]
    pos_label = cfg["pos_label"]

    # precision (PPV), recall y f1
    if average == "binary":
        prec = precision_score(y, y_pred, average=average, pos_label=pos_label, zero_division=0)
        rec  = recall_score(y, y_pred, average=average, pos_label=pos_label, zero_division=0)
        f1   = f1_score(y, y_pred, average=average, pos_label=pos_label, zero_division=0)
    
    else:
        prec = precision_score(y, y_pred, average=average, zero_division=0)
        rec  = recall_score(y, y_pred, average=average, zero_division=0)
        f1   = f1_score(y, y_pred, average=average, zero_division=0)

    return {
        "dataset": dataset_name,
        "precision": float(prec),             # PPV
        "balanced_accuracy": float(bacc),
        "recall": float(rec),
        "f1_score": float(f1),
    }

def ensure_output_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

import json
def main():
    # Modelo
    model_path = find_model_path()
    model = load_model(model_path)

    # Métricas
    train_metrics = compute_metrics(model, X_train, y_train, "train")
    test_metrics  = compute_metrics(model, X_test,  y_test,  "test")

    # Guardado en JSON Lines (una fila por diccionario)
    ensure_output_dir(OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")
        f.write(json.dumps(test_metrics,  ensure_ascii=False) + "\n")

    print(f"Métricas guardadas en: {OUTPUT_JSON}")
    print("Ejemplo de filas escritas:")
    print(train_metrics)
    print(test_metrics)


STRICT_EXAMPLE_TYPO = False  
TYPO_LABEL = "1" 


def build_cm_entry(y_true, y_pred, dataset_name):
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    entry = {"type": "cm_matrix", "dataset": dataset_name}

    for i, true_lbl in enumerate(labels):
        row = {}
        for j, pred_lbl in enumerate(labels):
            pred_lbl_str = str(pred_lbl)
            if STRICT_EXAMPLE_TYPO and pred_lbl_str == TYPO_LABEL:
                pred_key = f"predicte_{pred_lbl_str}" 
            
            else:
                pred_key = f"predicted_{pred_lbl_str}"
            row[pred_key] = int(cm[i, j])
        entry[f"true_{str(true_lbl)}"] = row

    return entry

def main_2():
    # Modelo
    model_path = find_model_path()
    model = load_model(model_path)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Entradas de matrices de confusión
    train_entry = build_cm_entry(y_train, y_pred_train, "train")
    test_entry  = build_cm_entry(y_test,  y_pred_test,  "test")

    # Append en JSON Lines
    ensure_output_dir(OUTPUT_JSON)
    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(train_entry, ensure_ascii=False) + "\n")
        f.write(json.dumps(test_entry,  ensure_ascii=False) + "\n")
    
    print(f"Matrices de confusión agregadas a: {OUTPUT_JSON}")
    print(train_entry)
    print(test_entry)

from sklearn.model_selection import ParameterSampler, StratifiedKFold, GridSearchCV

if __name__ == "__main__":
    # Rutas por defecto 
    ruta_train = "files/input/train_data.csv/train_default_of_credit_card_clients.csv"
    ruta_test = "files/input/test_data.csv/test_default_of_credit_card_clients.csv"

    df_train, df_test = cargar_y_procesar(ruta_train, ruta_test)
    X_train, y_train, X_test, y_test =dividir_en_xy(df_train, df_test)
    print("Listo lindos")

    # 2) Definir columnas categóricas y numéricas
    cat_cols_explicit = ['SEX', 'EDUCATION', 'MARRIAGE'] + [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    categorical_cols = [c for c in cat_cols_explicit if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    pipe = crear_pipeline_rf(categorical_cols, numeric_cols)

   
    from sklearn.model_selection import ParameterSampler, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from tqdm import tqdm
    import numpy as np

  
    param_dist = {
    "clf__C": np.logspace(-3, 3, 7),       
        "clf__gamma": ["scale", "auto"],         
        "clf__kernel": ["rbf"],                 
    }

    # Generar combinaciones aleatorias 
    n_iter = 50
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    
    param_grid = [
        {k: v if isinstance(v, (list, tuple, np.ndarray)) else [v] for k, v in combo.items()}
        for combo in param_list
    ]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,               
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,                         
        verbose=2,
        return_train_score=False,
        error_score="raise"                  
    )

    print(f"🔎 Probando {n_iter} combinaciones con 10-fold CV (GridSearchCV)...")
    grid.fit(X_train, y_train)

    print("\nMejor balanced accuracy (CV): {:.4f}".format(grid.best_score_))
    print("Mejores hiperparámetros:", grid.best_params_)

    
    # A partir de aquí, 'model' ES GridSearchCV
    model = grid

    # Guardar el objeto GridSearchCV completo
    import pickle, gzip
    from pathlib import Path

    model_dir = Path("files/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl.gz"

    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Modelo (GridSearchCV) guardado en: {model_path}")

    # Verificación explícita de tu requerimiento
    print(str(type(model)))  # debería contener 'GridSearchCV'
    assert "GridSearchCV" in str(type(model))
    
    MODEL_PATHS = [
    "files/models/model.pkl.gz"
    ]   

    OUTPUT_JSON = "files/output/metrics.json"
    main()
    main_2()
