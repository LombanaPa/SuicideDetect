route_project = '/Users/plombana/Documents/Universidad/ProyectodeGrado/SuicideDetect'
route_dataset = 'src/app/data/datasets/'
route_results = '/Users/plombana/Documents/Universidad/ProyectodeGrado/SuicideDetect/results/'
name_file = 'dataset_train_spanish.parquet'
name_clean_file = 'dataset_cleaned.parquet'
dataframe_embeddings = 'embeddings.csv'
save_train_results = 'curve_train.csv'
name_model_bert = 'model_ber_copia.h5'
ROUTES = dict(
    route_project = route_project,
    route_dataset = route_dataset,
    name_file = name_file,
    name_clean_file = name_clean_file,
    dataframe_embeddings = dataframe_embeddings,
    route_results = route_results
)

MODEL_EMBEDDINGS = 'sentence-transformers/LaBSE'

USEFULL_COLUMNS = ['Label', 'text_clean', 'embeddings']
MODEL_COLUMNS = ['Label', 'embeddings']

MODEL_PARAMETERS = {
    'learning_rate' : 0.01,
    'batch_size': 128,
    'epoch': 100,
    'dropout': 0.2,
     'validation_split':0.2,
    'shape':768
}

TRAINING_RESULTS = dict(
    save_train_results = save_train_results,
    name_model_bert = name_model_bert

)