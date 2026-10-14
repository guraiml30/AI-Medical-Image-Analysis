"""
main.py - command-line entry point for individual steps.

    python main.py simulate            # create virtual dataset in data/
    python main.py train [--epochs N] [--transfer]
    python main.py evaluate            # metrics + confusion matrix
    python main.py predict path/to/xray.png
    python main.py api [--port 8000]   # themed web app

To run EVERYTHING in one go use:  python rullall.py
"""
import argparse

from src import config


def main():
    ap = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis System")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("simulate")
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=config.EPOCHS); t.add_argument("--transfer", action="store_true")
    sub.add_parser("evaluate")
    p = sub.add_parser("predict"); p.add_argument("image")
    a = sub.add_parser("api"); a.add_argument("--host", default="127.0.0.1"); a.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    config.ensure_dirs()

    if args.cmd == "simulate":
        from src.simulate_data import generate_dataset
        print("Generated simulated dataset in", generate_dataset())
    elif args.cmd == "train":
        from src.data_loader import load_split
        from src.train import train
        X, y = load_split("train")
        train(X, y, epochs=args.epochs, transfer=args.transfer)
    elif args.cmd == "evaluate":
        from src.data_loader import load_split
        from src.evaluate import evaluate
        from src.train import load_trained
        X, y = load_split("test")
        m, *_ = evaluate(load_trained(), X, y)
        print(m)
    elif args.cmd == "predict":
        from src.predict import predict_file
        from src.train import load_trained
        print(predict_file(load_trained(), args.image))
    elif args.cmd == "api":
        from src.api import run_api
        run_api(args.host, args.port)


if __name__ == "__main__":
    main()
