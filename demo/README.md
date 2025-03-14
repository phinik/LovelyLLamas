# QUICK INSTRUCTIONS
Backend:
[The files neeeded for the backend will either already be there or linked otherwise]
1. Create backend/assets/ folder which will contains cities.json and dset_test.json
2. Create backend/data which will contain the dataset with all JSON files for the cities. Simply extract one of the dataset zip files to this location.
3. Backend/data must contain the following JSON files:
    - context_tokens
    - dset_test
    - rep_short_tokens_bert
    - target_tokens
   as well as the data folder from the dataset.
4. Create backend/src_model/ folder which will contain the model files:
    - best_model_CE_loss.pth
    - best_model_CE_loss_lstm.pth
    - params.json
    - params_lstm.json
5. Backend written with Flask, the app.py File contains import errors, since the Docker container copies the file into the correct directory
6. The Dockerfile contains the necessary commands to build the Docker container
7. Use docker compose up --build from root to assemble demo
8. access the demo at localhost:80
    
