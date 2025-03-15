# Repository Structure
Our repository is structured as follows:

- checkpoints => **Don't touch**. Contains the two custom classifier models that are needed for evaluating the respective metrics.
- data => everything related to the creation of the datasets, from data scraping, over EDA, post-processing, ChatGPT rewriting to dataset creation
- datasets => the two datasets as zip files
- demo => the demo web app as a docker compose
- src => our DL pipeline as it was used to train the transformers and the LSTMs
- tests => some tests for the DL pipeline using the unittest library

# How to install our repository
We are using [Poetry](https://python-poetry.org/) for dependency management. Hence, to get our code running, you need to perform the following steps:
1. Install Poetry if you have not already. Click [HERE](https://python-poetry.org/docs/#installing-with-the-official-installer) if you need a tutorial.
2. Clone our repository, cd into it and run `poetry install` in the root directory of our repository.

# Installing a dataset
Our two datasets are shipped with the repo, though as zip files in order to speed up the download. The zip files are contained in the `datasets` directory. To use any of the two, simply unzip the zip file to a location of your choice. `dataset_2024_12_12_wettercom.zip` contains the data collected from wetter.com, whereas `dataset_2024_12_12_chatGPT` contains the weather reports that were rewritten with the help of ChatGPT.

# Models included in the repository
We provide you with some of our models so that you can test them without having to train a new model:
- final_dmodel_64_2024_12_12_bert_ctc: Medium CTC Transformer using the Bert tokenizer. Trained on the wetter.com dataset. To be used with the `src/*_transformer.py` files.
- test_dmodel_256_2024_12_12_bert_ct_apo: Huge CT Transformer using the Bert tokenizer. Trained on the ChatGPT dataset. To be used with the `src/*_transformer.py` files.


# How to run our models
Before running any of our models, make sure you have activated the poetry virtual environment. For this, cd into our repository and simply type `poetry shell`. Now you are good to go.

## Train a model
Transformers are trained using `src/train_transformer.py`. The following parameters are available:
1. name: Name of the run
2. dataset_path: Path to dataset root
3. checkpoints_path: Where to store checkpoints
4. tensorboard_path: Where to store tensorboard summary
5. model: Which model to use, choices are "og_transformer" (default transformer), "rope_transformer", "full_rope_transformer"
6. cache_data: All data will be loaded into the RAM before training
7. tokenizer: Which tokenizer to use for the weather report, choices are "sow" and "bert"
8. model_config: What transformer model configuration to use. This refers to the files in `src/transformer_configs`.
9. num_workers: How many workers to use for data loading
10. target: What to train on, choices are "default" (wetter.com) and "gpt". Note that the respective dataset must be used!
11. overview: What context to use, choices are "full", "ctpc", "ctc", "ct" and "tpwc"
12. num_samples: How many samples to use during training, choices are -1 (all), 100, 200, 400, 800, 1600, 3200, 6400

Other parameters, such as the number of epochs, must be changed within the file itself, i. e. at the bottom of the file.


For example, a command to train a default, non-RoPE transformer of Medium size (d_model=64) on the wetter.com dataset using the Bert tokenizer, a CT context and 6400 samples could look like:

    python src/train_transformer.py --dataset_path ~/dataset_2024_12_12_wettercom --checkpoints_path ./checkpoints --model og_transformer --cache_data --tensorboard_path ./tensorboard --tokenizer bert --target default --name final_dmodel_64_2024_12_12_bert_ct_6400 --overview ct --model_config src/transformer_configs/dmodel_64_tiny.json --num_samples 6400

Do not get confused with the naming of the model config files. The suffixes "tiny", "small" and "big" originate from the intermediate presentation, and were kept for backwards compatibility with the models from that time. They are **not** the same as the names used in the final report. The final report uses the following naming convention:
- tiny: dmodel_16_tiny.json
- small: dmodel_32.json
- medium: dmodel_64_tiny.json
- big: dmodel_128_small.json
- huge: dmodel_256_big.json

By default, the model is saved after each epoch and the best model is saved in a separate file, called `best_model_CE_loss`, in the checkpoints path. Some metadata is saved there too. You should not mess around with these files as they are needed to automatically configure the evaluation and text generation.


The custom classifiers are trained in a similar fashion using `src/train_classifier.py`. Note, however, that the arguments are slighly different. Run
`src/train_classifier.py -h` in order to get an overview over the arguments. The `-h` argument works for any training, evaluation or generation file.


## Evaluate a model
Transformers are evaluated using `src/eval_metrics_transformer.py`. The following parameters are available:
1. dataset_path: Path to dataset root. This should be equal to the path used for the training of the model.
2. model_weights: Which model weights to use
3. metrics: Select which metrics shall be computed, choices are "bertscore", "bleu", "rouge", "temps" (Temp_Ghost), "temp_range", "cities", "classifier" and "classifier_ct". Note: 'classifier' and 'classifier_ct' do not work with SoW models!
4. output_filename: If output shall be saved to a different file than the standard file. By default, the results are saved to `eval_<model_weights>.json` in the respective checkpoints directory.


For example, in order to evaluate the best model of the above training, we could run:

    python src/eval_metrics_transformer.py --dataset_path ~/dataset_2024_12_12_wettercom --metrics temps temp_range cities classifier classifier_ct --model_weights checkpoints/final_dmodel_64_2024_12_12_bert_ct_6400/best_model_CE_loss.pth

This would generate a file `eval_best_model_CE_loss.json` in `checkpoints/final_dmodel_64_2024_12_12_bert_ct_6400` containing the values for the selected metrics.

## Generate from a model
To generate from a transformer, use `src/generate_transformer.py`. The following parameters are available:
1. dataset_path: Path to dataset root
2. model_weights: Which model weights to use

Again, we could generate from our newly trained transformer using:

    python src/generate_transformer.py --dataset_path ~/dataset_2024_12_12_wettercom --model_weights checkpoints/final_dmodel_64_2024_12_12_bert_ct_6400/best_model_CE_loss.pth

By default, this will generate weather reports for 10 samples from the test dataset and print them to the terminal along the respective target.

