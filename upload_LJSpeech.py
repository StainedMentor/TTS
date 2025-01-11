from clearml import Dataset

def create_ljspeech_dataset(data_dir="../LJSpeech-1.1"):
    """
    Creates a ClearML dataset from the LJSpeech-1.1 folder.
    """
    ljspeech_dataset = Dataset.create(
        dataset_name="LJSpeech-1.1",
        dataset_project="TTS Project",
    )
    ljspeech_dataset.add_files(path=data_dir)
    ljspeech_dataset.get_logger().report_text(
        "LJSpeech 1.1 dataset added: wavs + metadata.csv"
    )

    ljspeech_dataset.upload()
    ljspeech_dataset.finalize()
    print("LJSpeech-1.1 dataset successfully created in ClearML.")


if __name__ == "__main__":
    create_ljspeech_dataset()
