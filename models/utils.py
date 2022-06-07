from models.custom_models import electra

def get_model(model_name:str, pretrained):
    
    if model_name == 'electra':
        return electra(pretrained)

if __name__ == '__main__':
    pass