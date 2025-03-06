import os
import yaml

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        """
        Inizializza il caricamento della configurazione YAML.
        
        :param config_path: Percorso del file di configurazione (relativo o assoluto).
        """
        self.config_path = config_path
        self.config = self.load_config()

    def resolve_paths_in_config(self, config):
        """
        Risolve i percorsi relativi nei percorsi assoluti basati sulla directory del file di configurazione.
        
        :param config: La configurazione caricata dal file YAML.
        :return: La configurazione con i percorsi risolti.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory del file corrente
        for key, value in config["paths"].items():
            if isinstance(value, str) and value.startswith("../"):  # Solo se è un percorso relativo
                # Risolvi il percorso relativo in assoluto
                config["paths"][key] = os.path.abspath(os.path.join(base_dir, value))
        return config

    def load_config(self):
        """
        Carica e risolve i percorsi nel file di configurazione YAML.
        
        :return: La configurazione con i percorsi risolti.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Risolvi i percorsi relativi
        config = self.resolve_paths_in_config(config)

        return config

    def get_config(self):
        """
        Restituisce la configurazione risolta.
        
        :return: La configurazione con i percorsi risolti.
        """
        return self.config