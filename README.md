# lstm_sequence_classification_csci467

This is the semester project for a machine learning class I took as an undergrad in 2023. It compares various models including LSTM and fine-tuned DistilBERT on sentiment analysis. Report.pdf has a very detailed paper-like presentation of what was done. 

Each file has one of the three models. I highly recommend training them on google Collab.
In the report you can find the link to the google drive that contains the necessary data i.e. the splits that were generated
with the create_dataset.py quick script and the glove word embeddings.

To keep the code clear and because I used google collab, I didn't use arguments. Instead everytime the code is run
all of the experiments are conducted (training, evaluating on dev set, evaluating on test set, print confusion matrices
and analyze mistakes). The hyperparameters are located at the top of each file (hidden size, embedding size etc.) and the
paths to the files of the data and word embeddings so they can be easily tweaked.
