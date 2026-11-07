import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="runs/visdrone_yolo11n/weights/best.pt"
    )

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    model = YOLO(args.weights)

    source = Path(args.source)

    if source.is_dir():

        images = [
            p for p in source.iterdir()
            if p.suffix.lower() in [
                ".jpg", ".jpeg", ".png"
            ]
        ]

    else:
        images = [source]

    output = Path("results/predictions")
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    for image in images:

        results = model.predict(
            source=str(image),
            conf=args.conf,
            device="cpu",
            imgsz=640,
            verbose=False
        )

        result = results[0]

        counts = {}

        for box in result.boxes:

            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            counts[class_name] = (
                counts.get(class_name, 0) + 1
            )

        result.save(
            filename=str(output / image.name)
        )

        rows.append({
            "imagem": image.name,
            "total": len(result.boxes),
            "contagem_por_classe": str(counts)
        })

        print(
            image.name,
            "->",
            len(result.boxes),
            "detecções"
        )

    with open(
        output / "counts.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "imagem",
                "total",
                "contagem_por_classe"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()