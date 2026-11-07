import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path("dataset/dataset_yolo")
PROCESSED = Path("dataset/processed")

VARIANTS = [
    "grayscale",
    "gaussian",
    "sobel",
    "equalized"
]


def main():

    executable = Path("cimg/build/cimg_preprocess")

    images = ROOT / "images/val"
    labels = ROOT / "labels/val"

    if not executable.exists():
        raise SystemExit("Compile o CImg primeiro: make -C cimg")

    subprocess.run(
        [
            str(executable),
            str(images.resolve()),
            str(PROCESSED.resolve())
        ],
        check=True
    )

    with open("visdrone.yaml", "r") as file:
        original_config = yaml.safe_load(file)

    for variant in VARIANTS:

        variant_root = ROOT / "variants" / variant

        images_out = variant_root / "images/val"
        labels_out = variant_root / "labels/val"

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image in (PROCESSED / variant).glob("*.png"):

            shutil.copy2(
                image,
                images_out / image.name
            )

            label = labels / f"{image.stem}.txt"

            if label.exists():
                shutil.copy2(
                    label,
                    labels_out / label.name
                )
            else:
                (labels_out / label.name).write_text("")

        config = {
            "path": str(variant_root.resolve()),
            "train": str((ROOT / "images/train").resolve()),
            "val": str(images_out.resolve()),
            "names": original_config["names"]
        }

        yaml_path = Path(f"visdrone_{variant}.yaml")

        with open(yaml_path, "w") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print("Criado:", yaml_path)


if __name__ == "__main__":
    main()