from email.policy import default
import click
from rich.progress import track
from librairies.utils import extract_features, train_model,predict


@click.group(chain=False, invoke_without_command=True)
def handle_command():
    pass


@handle_command.command(name="extraction", help='extraction of features')
@click.option('--path_to_data', help="put the path of the folder that contains your subfolders defining your classes", type=click.Path(True))
@click.option('--path_to_save', help="put the path where you want to save your dataframe.example: /home/cifope/dataframe/dataset.pkl")
def extraction(path_to_data, path_to_save):

    extract_features(path_to_data, path_to_save)


@handle_command.command(name="train", help='train your model')
@click.option('--path_to_dataset', help="put the path of your dataset file", type=click.Path(True))
@click.option('--path_to_save', help="put the path where you want to save your model.example: /home/cifope/checkpoints/model.pth")
@click.option('--batch_size', default=32, help='put the batch size')
@click.option('--epochs', default=64, help='put the number of epoch')
@click.option('--lr', default=1e-3, help='put your learning rate')
def train(path_to_dataset, path_to_save, batch_size, epochs, lr):

    train_model(path_to_dataset, path_to_save,
                batch_size, epochs, lr)
    
@handle_command.command(name="predict",help='do a prediction')
@click.option('--path_to_model',help='put the path to your model')
@click.option('--path_to_image',help='put the path to image')
def prediction(path_to_model,path_to_image):
    predict(path_to_model,path_to_image)


if __name__ == "__main__":
    handle_command()
