import os
import yaml

def get_model_name(config: dict) -> str:
    model_params = config['model_params']
    data_params = config['data_params']
    model_name = config['model_class']

    other_params = data_params
    model_parameter_string = '_'.join([str(value) for key, value in model_params.items()])

    trainer_parameter_string = '&'.join([ str(key) +"="+ str(value) for key, value in other_params.items()])
    return model_name + '(' + model_parameter_string + ')' + '?' + trainer_parameter_string

def load_config(config_name: str) -> dict:
    file_name = config_name + '.yaml'
    
    for folder in os.listdir('configs'):
        if os.path.isfile('configs/' + file_name):
            return yaml.load(open('configs/' + file_name, 'r'), Loader=yaml.FullLoader)
        
    raise FileNotFoundError('Config file not found')