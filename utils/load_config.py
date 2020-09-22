import ruamel.yaml
import os


class LoadConfig:
    def __init__(self, config_path):
        self.yaml = ruamel.yaml.YAML()
        self.config_path = config_path

    def __call__(self):
        with open(self.config_path+'/data_config.yaml', 'rb') as config:
            config_dict = self.yaml.load(config)
        return dict(config_dict)




if __name__ == '__main__':
    config_loader = LoadConfig('./config')

    config_loader
    config = config_loader()
    print(config)
