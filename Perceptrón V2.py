import csv
from pathlib import Path

import numpy as np


# 1) Datos de entrada y salidas para cada compuerta
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=float,
)

GATES = {
    "OR": np.array([0, 1, 1, 1], dtype=float),
    "AND": np.array([0, 0, 0, 1], dtype=float),
}


# 2) Hiperparámetros para las pruebas
learning_rate = 0.1
epochs_list = [100, 1000]
threshold_list = [0.0]
error_tolerance = 0.01
base_seed = 42


# 3) Se agrega sesgo como una columna de unos
Xb = np.hstack([X, np.ones((X.shape[0], 1))])


# 4) Funciones de activación
def step(z, threshold=0.0):
    return 1.0 if z >= threshold else 0.0


def sign(z, threshold=0.0):
    return 1.0 if z >= threshold else -1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def linear(z):
    return z


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def train_and_evaluate(X_bias, y, activation_name, threshold, epochs, lr, tol, seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.5, 0.5, size=(X_bias.shape[1],))

    if activation_name == "sign":
        y_train = np.where(y == 0, -1.0, 1.0)
    else:
        y_train = y.copy()

    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X_bias, y_train):
            z = float(np.dot(xi, w))

            if activation_name == "step":
                y_out = step(z, threshold)
                delta = yi - y_out
                should_update = delta != 0
            elif activation_name == "sign":
                y_out = sign(z, threshold)
                delta = yi - y_out
                should_update = delta != 0
            elif activation_name == "sigmoid":
                y_out = sigmoid(z)
                delta = yi - y_out
                should_update = abs(delta) > tol
            elif activation_name == "linear":
                y_out = linear(z)
                delta = yi - y_out
                should_update = abs(delta) > tol
            else:
                raise ValueError(f"Activación no soportada: {activation_name}")

            if should_update:
                w += lr * delta * xi
                errors += 1

        if errors == 0:
            break

    y_pred = []
    for xi in X_bias:
        z = float(np.dot(xi, w))
        if activation_name == "step":
            y_hat = step(z, threshold)
        elif activation_name == "sign":
            y_hat = (sign(z, threshold) + 1.0) / 2.0
        elif activation_name == "sigmoid":
            y_hat = sigmoid(z)
        else:
            y_hat = linear(z)
        y_pred.append(y_hat)

    y_pred = np.array(y_pred, dtype=float)
    return mse(y, y_pred), r2_score(y, y_pred)


def export_results(results, output_xlsx, output_csv):
    headers = [
        "Compuerta",
        "ECM",
        "R²",
        "Umbral",
        "Épocas",
        "Escalera",
        "Signo",
        "Sigmoidea",
        "Lineal",
    ]

    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Comparativa"
        ws.append(headers)
        for row in results:
            ws.append([row[h] for h in headers])

        wb.save(output_xlsx)
        print(f"Archivo Excel generado: {output_xlsx}")
    except Exception:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print("No se encontró openpyxl para crear .xlsx.")
        print(f"Se generó CSV compatible con Excel: {output_csv}")


def main():
    activations = ["step", "sign", "sigmoid", "linear"]
    rows = []

    for gate_name, y in GATES.items():
        for threshold in threshold_list:
            for epochs in epochs_list:
                for activation in activations:
                    score_mse, score_r2 = train_and_evaluate(
                        X_bias=Xb,
                        y=y,
                        activation_name=activation,
                        threshold=threshold,
                        epochs=epochs,
                        lr=learning_rate,
                        tol=error_tolerance,
                        seed=base_seed,
                    )

                    rows.append(
                        {
                            "Compuerta": gate_name,
                            "ECM": round(score_mse, 6),
                            "R²": round(score_r2, 6),
                            "Umbral": threshold,
                            "Épocas": epochs,
                            "Escalera": "X" if activation == "step" else "",
                            "Signo": "X" if activation == "sign" else "",
                            "Sigmoidea": "X" if activation == "sigmoid" else "",
                            "Lineal": "X" if activation == "linear" else "",
                        }
                    )

    here = Path(__file__).resolve().parent
    output_xlsx = here / "tabla_perceptron_resultados.xlsx"
    output_csv = here / "tabla_perceptron_resultados.csv"
    export_results(rows, output_xlsx, output_csv)

    by_activation = {}
    for activation in activations:
        activation_label = {
            "step": "Escalera",
            "sign": "Signo",
            "sigmoid": "Sigmoidea",
            "linear": "Lineal",
        }[activation]

        subset = [r for r in rows if r[activation_label] == "X"]
        avg_mse = float(np.mean([r["ECM"] for r in subset]))
        avg_r2 = float(np.mean([r["R²"] for r in subset]))
        by_activation[activation_label] = {"ECM": avg_mse, "R²": avg_r2}

    best = min(by_activation.items(), key=lambda item: (item[1]["ECM"], -item[1]["R²"]))
    print("Resumen promedio por activación:")
    for name, metrics in by_activation.items():
        print(f"  {name}: ECM={metrics['ECM']:.6f}, R²={metrics['R²']:.6f}")
    print(f"Mejor desempeño global: {best[0]}")


if __name__ == "__main__":
    main()


