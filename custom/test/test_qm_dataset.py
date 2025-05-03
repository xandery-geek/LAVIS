from custom.datasets.builders.qm_builder import QMRetrievalBuilder


if __name__ == "__main__":
    # Example usage
    builder = QMRetrievalBuilder()
    datasets = builder.build()

    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    val_glance = val_dataset.__getitem__(0)
    test_glance = test_dataset.__getitem__(0)

    print("Validation dataset sample:", val_glance)
    print("Test dataset sample:", test_glance)
