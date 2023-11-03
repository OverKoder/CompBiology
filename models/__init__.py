from .vae import VAE


# here we can add other models
MODELS_LIST = [ VAE]

def get_model_class(model_class_name: str):
    for model in MODELS_LIST:
        if model.__name__ == model_class_name:
            return model
    raise ValueError(f"Model {model_class_name} not found")


