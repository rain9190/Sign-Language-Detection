# Sign Language Recognition using SVM and LSTM

Classifying sign gestures from Leap Motion recordings. Compares an SVM with an LSTM across 42 classes and includes a demo using 28 sequences of recorded words.

## Dataset

- 378 training files, with 9 files per class
- 90 frames per recording and 36 features per frame
- `Dataset/` contains the training recordings
- `sentence_data2/` contains the word recordings used in the demo

The notebook works with saved recordings. A Leap Motion sensor is not needed to run it.

## Approach

The recordings are parsed, standardized and split into training and test sets using a stratified 70/30 split.

- **SVM:** Uses the flattened recordings with an RBF kernel.
- **LSTM:** Uses 120 LSTM units, dropout of 0.3, a dense layer with 100 units and a softmax output layer. Trained for 50 epochs with Adam and a batch size of 32.

The LSTM uses 20% of the training split for validation. Scaling is fitted only on the LSTM fitting subset and applied to both models.

## Results

Results from the updated notebook:

| Metric | SVM | LSTM |
| --- | ---: | ---: |
| Test accuracy | 92.98% | 93.86% |
| Test macro-F1 | 0.9228 | 0.9251 |
| Demo mean word accuracy | 93.63% | 97.20% |

The demo predicts individual words in predefined sequences. Its score is the average word accuracy across those sequences, not complete-sentence accuracy.

## Running the notebook

Use Python 3.11. From the project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `Sign_Lang_Detection.ipynb`, select the virtual environment as the kernel, and run all cells. Keep `Dataset/` and `sentence_data2/` in the same folder as the notebook.

## Future Work

Extending the project to recognize signs live through a webcam by collecting camera-based training data and retraining the model.