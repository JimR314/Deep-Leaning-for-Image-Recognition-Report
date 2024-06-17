import os
import json
from tensorflow.keras.models import load_model # type: ignore

def save_model_and_history(model, history, model_name, results_dir='results'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    model_path = os.path.join(results_dir, model_name + '.keras')
    history_path = os.path.join(results_dir, model_name + '_history.json')
    
    model.save(model_path)
    
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    print(f'Model and history saved to {results_dir}')

def load_model_and_history(model_name, results_dir='results'):
    model_path = os.path.join(results_dir, model_name + '.keras')
    history_path = os.path.join(results_dir, model_name + '_history.json')
    
    model = load_model(model_path)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, history
