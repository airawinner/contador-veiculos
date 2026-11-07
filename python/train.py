
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Ajuste fino do detector de veículos."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs/detect/vehicle_30_epochs/weights/last.pt"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args()

    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Pesos não encontrados: {weights}"
        )

    if not Path("visdrone.yaml").exists():
        raise FileNotFoundError(
            "visdrone.yaml não encontrado na raiz do projeto."
        )

    print("=" * 50)
    print("AJUSTE FINO DO DETECTOR DE VEÍCULOS")
    print("=" * 50)
    print(f"Pesos iniciais: {weights}")
    print(f"Épocas adicionais: {args.epochs}")
    print(f"Resolução: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print("Dispositivo: CPU")
    print("=" * 50)

    # Carrega os pesos do treinamento anterior.
    model = YOLO(str(weights))

    # Continua o aprendizado a partir dos pesos existentes.
    # Um novo otimizador será inicializado.
    model.train(
        data="visdrone.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,

        device="cpu",

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        patience=8,
        cos_lr=True,
        close_mosaic=5,

        project="runs/detect",
        name="vehicle_counting_final",
        exist_ok=True,

        pretrained=True,
        plots=True,
        save=True,
        seed=42,
        deterministic=True
    )

    print("\nTreinamento finalizado!")
    print(
        "Melhores pesos: "
        "runs/detect/vehicle_counting_final/weights/best.pt"
    )


if __name__ == "__main__":
    main()
