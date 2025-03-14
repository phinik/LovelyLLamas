# QUICK INSTRUCTIONS
Backend:
[The files neeeded for the backend will either already be there or linked otherwise]
1. Create backend/assets/ folder which will contains cities.json and dset_test.json
2. Create backend/data which will contain the dataset with all JSON files for the cities
3. Backend/data also contains following JSON files:
    - bags_of_words_gpt
    - bags_of_words
    - context_tokens
    - dset_test
    - rep_short_tokens_bert
    - target_tokens
4. Create backend/src_model/ folder which will contain the model files:
    - best_model_CE_loss.pth
    - best_model_CE_loss_lstm.pth
    - params.json
    - params_lstm.json
5. Backend written with Flask, the app.py File contains import errors, since the Docker container copies the file into the correct directory
6. The Dockerfile contains the necessary commands to build the Docker container
7. Use docker compose up --build from root to assemble demo
8. access the demo at localhost:80
    