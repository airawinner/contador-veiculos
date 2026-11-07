
import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml


CATEGORY_MAP = {
    3: 0,   # bicycle
    4: 1,   # car
    5: 2,   # van
    6: 3,   # truck
    7: 4,   # tricycle
    8: 5,   # awning-tricycle
    9: 6,   # bus
    10: 7   # motor
}

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

# Limite para manter o experimento rápido
MAX_IMAGES = 200

RANDOM_SEED = 42


def convert_box(x, y, w, h, image_width, image_height):

    x1 = max(0, min(image_width, x))
    y1 = max(0, min(image_height, y))

    x2 = max(0, min(image_width, x + w))
    y2 = max(0, min(image_height, y + h))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2) / image_width
    cy = ((y1 + y2) / 2) / image_height

    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height

    return cx, cy, bw, bh


def main():

    parser = argparse.ArgumentParser(
        description="Converte o VisDrone para o formato YOLO."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Pasta VisDrone2019-DET-train"
    )

    parser.add_argument(
        "--output",
        default="dataset/dataset_yolo"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
        help="Quantidade máxima de imagens utilizadas."
    )

    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio deve estar entre 0 e 1.")

    if args.max_images < 1:
        raise ValueError("--max-images deve ser maior que zero.")

    source = Path(args.source)
    output = Path(args.output)

    images_dir = source / "images"
    annotations_dir = source / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {images_dir}"
        )

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Pasta de anotações não encontrada: {annotations_dir}"
        )

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    if not images:
        raise RuntimeError("Nenhuma imagem encontrada.")

    # Seleciona uma amostra reproduzível
    random.seed(RANDOM_SEED)
    random.shuffle(images)

    images = images[:args.max_images]

    split = int(len(images) * (1 - args.val_ratio))

    datasets = {
        "train": images[:split],
        "val": images[split:]
    }

    print(f"Imagens selecionadas: {len(images)}")
    print(f"Treinamento: {len(datasets['train'])}")
    print(f"Validação: {len(datasets['val'])}")

    total_invalid = 0
    total_objects = 0

    for name, image_list in datasets.items():

        image_output = output / "images" / name
        label_output = output / "labels" / name

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        for image_path in image_list:

            annotation_path = (
                annotations_dir /
                f"{image_path.stem}.txt"
            )

            image = cv2.imread(str(image_path))

            if image is None:
                print("Imagem inválida:", image_path)
                total_invalid += 1
                continue

            height, width = image.shape[:2]

            shutil.copy2(
                image_path,
                image_output / image_path.name
            )

            labels = []

            if annotation_path.exists():

                with open(annotation_path, "r") as file:

                    for line in file:

                        values = line.strip().split(",")

                        if len(values) < 8:
                            continue

                        try:
                            x, y, w, h = map(
                                float, values[:4]
                            )

                            score = int(float(values[4]))
                            category = int(float(values[5]))

                        except ValueError:
                            continue

                        if score == 0:
                            continue

                        if category not in CATEGORY_MAP:
                            continue

                        box = convert_box(
                            x, y, w, h,
                            width, height
                        )

                        if box is None:
                            continue

                        class_id = CATEGORY_MAP[category]

                        labels.append(
                            f"{class_id} " +
                            " ".join(
                                f"{v:.6f}" for v in box
                            )
                        )

            total_objects += len(labels)

            label_file = (
                label_output /
                f"{image_path.stem}.txt"
            )

            label_file.write_text(
                "\n".join(labels),
                encoding="utf-8"
            )

    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            i: name
            for i, name in enumerate(CLASS_NAMES)
        }
    }

    with open("visdrone.yaml", "w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print("\n================================")
    print("DATASET PREPARADO")
    print("================================")
    print("Imagens selecionadas:", len(images))
    print("Treinamento:", len(datasets["train"]))
    print("Validação:", len(datasets["val"]))
    print("Objetos anotados:", total_objects)
    print("Imagens inválidas:", total_invalid)
    print("Configuração: visdrone.yaml")
    print("================================")


if __name__ == "__main__":
    main()
