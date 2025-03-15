# Design behind the GRU

There was 3 GRU implementations developed as part of this project:
1. Basic GRU -> a model to create long reports, which was a personal side goal to earn knowledge.
2. GRU with Attention -> a model to create short reports (stylised), which was the main goal of the project.
3. Advanced GRU without Attention -> a model to create short reports (stylised) while not using attention.
4. The results vary a lot to the LSTM due to the GRU being made as to produce more complex reports (self-set challenge)

# Pipe
1. The Basic implementation has a unique pipeline with full feature set and NER tagging/replacement, while the other two share a pipeline with a reduced feature set for ease of learning and parameter reduction. 
2. The pipeline can be seen under weather_dataset.py. 
3. StandardWeatherDataset is the pipe for the Basic GRU, while SimpleWea(...) is the pipe for the other two. There is a ground class called BaseWea(...) that is inherited by both pipes as a means to share common code.

# Data
For the Pipes to work the JSONS containing the city weather data must be placed under data/files_for_chatGPT/2024-12-12/ folder.

# Text Generation
1. The models will be available through following link to download: https://drive.google.com/drive/folders/1knoes-q2vM1Bpen22mbbOnFyKL2Jhhi1?usp=sharing
2. After download place them in gru/models/ folder.
3. Use the generate.py script to generate text from the models.
4. Be careful to select correct model_type and model_path in script, if they don't match the model will not load.
5. Personally I wouldn't recommend using the Basic GRU, as the NER tagging takes approx. 30-40 mins based on a 12 core CPU.

# Training
1. The models were trained using basic_GRU_training.ipynb and attention_advanced_GRU_training.ipynb.