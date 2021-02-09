# On local machine
import fiftyone as fo

name = "my-images-dir"
dataset_dir = "./datas/raw_dataset/"

dataset = fo.Dataset.from_dir(
    dataset_dir, fo.types.ImageClassificationDirectoryTree, name=name
)


print(dataset)

session = fo.launch_app(dataset)
session.wait()
