import argparse, json
from anemia_model.predict import predict_one

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Ruta a archivo JSON con un caso")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    label, proba = predict_one(payload)
    out = {"prediccion": label, "probabilidades": proba}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
