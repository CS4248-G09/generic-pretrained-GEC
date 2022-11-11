## Pre-requisites
1. Install sentence_transformers: "conda install -c conda-forge sentence-transformers"
2. Install sklearn: "conda install -c anaconda scikit-learn"

## How To Run
1. Run the four models (baseline, cnn, mnli, xsum) to get their respective output
2. Place the outputs into the "Outputs" folder , with .txt at the back
3. Place a copy of the output from the best performing model into the "Best" folder inside "Outputs" folder 
4. Place the input into the Input folder, with .txt at the back
5. Type the command "python Ensemble.py flag" to get the aggregated output file(s)
    flag = -1 : Get the outputs from all our bs methods
    flag = 0 : Get the output from random aggregation
    flag = 1 : Get the output from max voting
    flag = 2 : Get the output from max voting at "token" level
    flag = 3 : Get the output from max voting with assumption 
    flag = 4 : Get the output from cosine similarity

