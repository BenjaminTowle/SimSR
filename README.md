# Model-based Simulation for Optimising Smart Reply
This is the code corresponding to the paper accepted to ACL 2023: Model-based Simulation for Optimising Smart Reply.

## Requirements
* torch
* transformers
* faiss
* datasets
* nltk
* rouge-score
* scipy

## Getting started
The code is designed to run on two datasets: Reddit and PERSONA-CHAT. Reddit can be downloaded from [here](https://github.com/zhangmozhi/mrs). Once downloaded, set the `FILE_PATH` and `SAVE_PATH` in `reddit.py` to the location where the train/test file can be found and where you want to save it respectively. Then run the file `reddit.py`. For PERSONA-CHAT, the data can be downloaded from [here](https://drive.google.com/open?id=1gNyVL5pSMO6DnTIlA9ORNIrd2zm8f3QH).

## Training
To train the code run the script `train.py`. Hyperparameters can be modified either in the `src/args.py` file, or by specifying the arguments directly when running the script. Note that you can either preprocess the dataset from the raw data obtained above, or can load an already processed dataset as a HuggingFace `DatasetDict` object. Assuming we are preprocessing the raw Reddit data, and training the base Matching model, we would do the following:
```
python train.py --output_dir PATH/TO/SAVE/MODEL \
    --data_dir PATH/TO/DATA/FOLDER \
    --dataset_save_path PATH/TO/SAVE/DATASET \
    --dataset_load_path none \
    --task reddit \
    --model_type matching \
    --bert_model_path distilbert-base-uncased
```

## Testing
To run predictions for SimSR on the test data you can call `test.py`. Note, the `response_set_path` argument determines the candidate pool used for retrieval. It is designed to load a `Dataset` object formatted the same as the datasets during training. Hence, you can use the path that points to the training dataset which can be found in the `DatasetDict` saved during training:
```
python test.py --model_load_path PATH/TO/TRAINED/MODEL \
    --response_set_path PATH/TO/REPLY/POOL \
    --agent_type simulation \
    --clustering exhaustive \
    --k 3
    --n 15
    --s 25
    --prediction_save_path PATH/TO/SAVE/PREDICTIONS
```

## Evaluation
We use the file `eval.py` for evaluating the predictions on the ROUGE and self-ROUGE metrics:
```
python eval.py --prediction_load_path PATH/TO/SAVE/PREDICTIONS
```

