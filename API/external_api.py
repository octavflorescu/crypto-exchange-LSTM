from flask import Flask, render_template, request
import os

from executor import *


app = Flask(__name__, static_folder='static')
PREDICT_SEQUENCE_FILENAME = 'predict_sequence'

@app.route("/")
@app.route("/home")
def result():
    return render_template('home.html')

@app.route("/train_predict", methods=['POST'])
def train_predict():
    all_file_names = [file_name for file_name in request.files.keys() if file_name != PREDICT_SEQUENCE_FILENAME]
    model_file_name = all_file_names[0] if all_file_names else None
    train_file = request.files[model_file_name]

    # download the file if needed
    train_file.save(model_file_name)

    # train if needed
    _exec = Executor(train_csv=model_file_name)
    model_saved_weights = f"{os.path.splitext(model_file_name)[0]}.pt"
    if os.path.isfile(model_saved_weights):
        # train was done on this train file
        _exec.train(with_plot=True)

    # predict if needed
    sequence_to_infer = request.files.get(PREDICT_SEQUENCE_FILENAME, None)
    if sequence_to_infer:
        # infer
        sequence_to_infer = pd.read_csv()[-_exec.seq_size:]
        output = _exec.infer(sequence_to_infer, load_from_file=model_saved_weights)
        return output
    else:
        return "training only"