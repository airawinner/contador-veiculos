
from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

# Caminhos
MODEL_PATH = Path(
    "runs/detect/runs/detect/vehicle_counting_final/weights/best.pt"
)

IMAGES_DIR = Path("dataset/dataset_yolo/images/val")
LABELS_DIR = Path("dataset/dataset_yolo/labels/val")
OUTPUT_DIR = Path("results/counting")

# Configurações
IMAGE_SIZE = 416
CONFIDENCE = 0.15
DEVICE = "cpu"

CLASS_NAMES = [
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_ground_truth(label_path):
    counts = np.zeros(len(CLASS_NAMES), dtype=int)

    if not label_path.exists():
        return counts

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if not values:
                continue

            class_id = int(values[0])

            if 0 <= class_id < len(CLASS_NAMES):
                counts[class_id] += 1

    return counts


def calculate_metrics(real, predicted):
    errors = predicted - real
    absolute_errors = np.abs(errors)

    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)

    total_real = np.sum(real)
    wape = (
        np.sum(absolute_errors) / total_real * 100
        if total_real > 0 else 0
    )

    exact = np.mean(errors == 0) * 100

    return mae, rmse, bias, wape, exact


def main():
    print("Carregando modelo...")
    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    real_counts = []
    predicted_counts = []
    rows = []

    print(f"\nAvaliando {len(image_paths)} imagens...\n")

    for index, image_path in enumerate(image_paths, 1):
        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        # Contagem real pelas anotações
        real = count_ground_truth(label_path)

        # Detecção pelo YOLO
        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )[0]

        predicted = np.zeros(len(CLASS_NAMES), dtype=int)

        if result.boxes is not None:
            for class_id in result.boxes.cls.cpu().numpy():
                class_id = int(class_id)

                if 0 <= class_id < len(CLASS_NAMES):
                    predicted[class_id] += 1

        real_total = int(np.sum(real))
        predicted_total = int(np.sum(predicted))

        real_counts.append(real)
        predicted_counts.append(predicted)

        rows.append({
            "imagem": image_path.name,
            "real": real_total,
            "previsto": predicted_total,
            "erro": predicted_total - real_total,
            "erro_absoluto": abs(predicted_total - real_total)
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}: "
            f"Real={real_total} | "
            f"Previsto={predicted_total}"
        )

    real_counts = np.array(real_counts)
    predicted_counts = np.array(predicted_counts)

    # Métricas gerais de contagem
    real_total = real_counts.sum(axis=1)
    predicted_total = predicted_counts.sum(axis=1)

    mae, rmse, bias, wape, exact = calculate_metrics(
        real_total, predicted_total
    )

    print("\n" + "=" * 50)
    print("RESULTADOS DA CONTAGEM")
    print("=" * 50)

    print(f"Imagens avaliadas: {len(image_paths)}")
    print(f"Veículos reais: {real_total.sum()}")
    print(f"Veículos previstos: {predicted_total.sum()}")
    print(f"MAE: {mae:.2f} veículos por imagem")
    print(f"RMSE: {rmse:.2f}")
    print(f"Viés médio: {bias:.2f}")
    print(f"WAPE: {wape:.2f}%")
    print(f"Contagem exata: {exact:.2f}% das imagens")

    # Métricas por classe
    print("\n" + "=" * 50)
    print("RESULTADOS POR CLASSE")
    print("=" * 50)

    for class_id, class_name in enumerate(CLASS_NAMES):
        real_class = real_counts[:, class_id]
        predicted_class = predicted_counts[:, class_id]

        if real_class.sum() == 0:
            continue

        c_mae, c_rmse, c_bias, c_wape, c_exact = (
            calculate_metrics(real_class, predicted_class)
        )

        print(f"\n{class_name}:")
        print(f"  Real: {real_class.sum()}")
        print(f"  Previsto: {predicted_class.sum()}")
        print(f"  MAE: {c_mae:.2f}")
        print(f"  WAPE: {c_wape:.2f}%")

    # Salvar comparação por imagem
    csv_path = OUTPUT_DIR / "counting_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "imagem",
                "real",
                "previsto",
                "erro",
                "erro_absoluto"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
