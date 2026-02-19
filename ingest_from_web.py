from learning_utils import import_from_web_manifest, fine_tune_from_dataset


if __name__ == "__main__":
    result = import_from_web_manifest("web_sources.json", data_dir="data")
    print("Downloaded:", result)

    stats = fine_tune_from_dataset("best_model.pth", data_dir="data", epochs=2)
    print("Model updated:", stats)
