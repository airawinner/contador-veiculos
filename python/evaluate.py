import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


VARIANTS = [
    ("original", "visdrone.yaml"),
    ("grayscale", "visdrone_grayscale.yaml"),
    ("gaussian", "visdrone_gaussian.yaml"),
    ("sobel", "visdrone_sobel.yaml"),
    ("equalized", "visdrone_equalized.yaml")
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="runs/visdrone_yolo11n/weights/best.pt"
    )

    parser.add_argument(
        "--all-variants",
        action="store_true"
    )

    parser.add_argument(
        "--data",
        default="visdrone.yaml"
    )

    args = parser.parse_args()

    model = YOLO(args.weights)

    experiments = (
        VARIANTS if args.all_variants
        else [("original", args.data)]
    )

    rows = []

    for name, yaml_path in experiments:

        if not Path(yaml_path).exists():
            print("Arquivo não encontrado:", yaml_path)
            continue

        print("\nExperimento:", name)

        metrics = model.val(
            data=yaml_path,
            imgsz=640,
            batch=1,
            device="cpu",
            workers=0,
            plots=True,
            project="runs/evaluation",
            name=name
        )

        rows.append({
            "experimento": name,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map)
        })

    if rows:

        output = Path("results/metrics")
        output.mkdir(parents=True, exist_ok=True)

        with open(
            output / "detection_metrics.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

        print("Resultados salvos!")


if __name__ == "__main__":
    main()