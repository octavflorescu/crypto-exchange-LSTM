from flask import Flask, render_template, request
import os
import sys
from executor import *


app = Flask(__name__, static_folder='static', static_url_path='/static')
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
    if not os.path.isfile(model_file_name):
        train_file.save(model_file_name)

    # train if needed
    _exec = Executor(train_csv=model_file_name)
    model_saved_weights = f"{os.path.splitext(model_file_name)[0]}.pt"
    print(f"Checking if model already trained at {model_file_name}", file=sys.stdout)
    if request.args.get('force_retrain', False) or not os.path.isfile(model_saved_weights):
        print(f"Started training and saving at {model_saved_weights}", file=sys.stdout)
        # train was done on this train file
        _exec.train(with_plot=True, epochs=int(request.args.get('epochs', 10)))

    # predict if needed
    sequence_to_infer = request.files.get(PREDICT_SEQUENCE_FILENAME, None)
    if sequence_to_infer:
        print(f"Started infering using model at {model_saved_weights}", file=sys.stdout)
        # infer
        sequence_to_infer = pd.read_csv(sequence_to_infer)[-_exec.seq_size:]
        output = _exec.infer(sequence_to_infer, load_from_file=model_saved_weights)
        return str(output)
    else:
        return "training only"

if __name__ == '__main__':
    app.run(debug=True, port=9998, host='0.0.0.0')