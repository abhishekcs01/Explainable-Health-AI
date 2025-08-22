
from __future__ import annotations
import argparse
import sys
from sklearn.preprocessing import MinMaxScaler

from xai_health.config import DATA_PATH, RANDOM_STATE
from xai_health import data as data_mod
from xai_health import features as F
from xai_health import model as M
from xai_health.explain import build_explainers
from xai_health.ui import build_interface

def _pipeline():
    df = data_mod.load_dataset(DATA_PATH)
    df = F.engineer(df)
    X, y = F.split_xy(df)

    scaler = F.build_scaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = M.make_splits(Xs, y)
    X_train, y_train = M.maybe_smote(X_train, y_train)

    model = M.train_model(X_train, y_train)
    return model, scaler, X_train, y_train, X_test, y_test

def cmd_train():
    model, scaler, X_train, y_train, X_test, y_test = _pipeline()
    print("✅ Training complete.")
    metrics = M.evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("Classification report:\n", metrics["report"])

def cmd_evaluate():
    model, scaler, X_train, y_train, X_test, y_test = _pipeline()
    metrics = M.evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("Classification report:\n", metrics["report"])
    fig = metrics["confusion_fig"]
    fig.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved confusion matrix to confusion_matrix.png")

def cmd_serve(host: str, port: int):
    model, scaler, X_train, y_train, X_test, y_test = _pipeline()
    shap_explainer, lime_explainer = build_explainers(model, X_train, F.FEATURES)
    app = build_interface(model, scaler, shap_explainer, lime_explainer)
    app.queue()
    app.launch(server_name=host, server_port=port, share=False, show_api=False)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Explainable AI for Heart Disease Risk")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train the model and print metrics")
    sub.add_parser("evaluate", help="Evaluate the model and save confusion matrix")

    pserve = sub.add_parser("serve", help="Run the Gradio app")
    pserve.add_argument("--host", default="127.0.0.1")
    pserve.add_argument("--port", default=7860, type=int)

    args = parser.parse_args(argv)

    if args.cmd == "train": cmd_train()
    elif args.cmd == "evaluate": cmd_evaluate()
    elif args.cmd == "serve": cmd_serve(args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
