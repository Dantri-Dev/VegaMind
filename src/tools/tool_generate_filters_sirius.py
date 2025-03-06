import groq
import pandas as pd
import json
import os
import logging
from src.config_loader import ConfigLoader

class ToolGenerateFiltersSirius:
    def __init__(self, config_path="config/config.yaml"):
        # Configura il logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Evita problemi con i tokenizer di Hugging Face
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Carica la configurazione utilizzando la classe ConfigLoader
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        print("Configurazione caricata correttamente.")
        
        self.cooking_techniques = self.load_cooking_techniques()
        
        print("[INIT] Inizializzazione del client Groq...\n")
        self.client = groq.Client(api_key=self.config["groq"]["api_key"])
        self.model = self.config["groq"]["model"]
        print("[OK] Client Groq inizializzato con il modello:", self.model, "\n")

    def load_cooking_techniques(self):
        """Carica le tecniche di cottura dal file CSV in un dizionario strutturato."""
        df = pd.read_csv(self.config["paths"]["tecniche_di_cottura"])

        # Creiamo un dizionario {Categoria: [Lista di Tecniche]}
        techniques_dict = df.groupby("Categoria")["Tecnica"].apply(list).to_dict()
        return techniques_dict
        
    def extract_techniques(self, user_query):
        """Recupera le tecniche di cottura dal Manuale di Sirius Cosmo."""
        techniques_by_category = {}

        # Scansioniamo tutte le categorie per vedere quali sono menzionate nella query
        for category, technique_list in self.cooking_techniques.items():
            if category.lower() in user_query.lower():
                techniques_by_category[category] = technique_list

        return techniques_by_category


    def get_restaurant_names(self, menus_dir):
        """Recupera i nomi dei ristoranti dai file PDF nella cartella dei menu."""
        try:
            restaurant_names = [
                os.path.splitext(f)[0]  # Rimuove l'estensione .pdf
                for f in os.listdir(menus_dir) 
                if f.endswith(".pdf")  # Considera solo i file PDF
            ]
            return restaurant_names
        except Exception as e:
            print(f"[ERRORE] Impossibile recuperare i nomi dei ristoranti: {e}")
            return []
        

    def generate_filters(self, user_query, techniques_by_category):
        """Genera filtri dinamici basati sulla richiesta dell'utente con distinzione tra AND e OR."""
        print("\n[STEP] Generazione dei filtri dinamici...\n")

        restaurant_names = self.get_restaurant_names(self.config["paths"]["menus_dir"])

        # Verifica che techniques_by_category sia un dizionario
        if not isinstance(techniques_by_category, dict):
            techniques_by_category = {"Generico": techniques_by_category}

        # Costruzione della stringa di tecniche (fuori dal loop per evitare sovrascrizioni)
        techniques_str = "\n".join([
            f"{category}: {', '.join(techniques)}"
            for category, techniques in techniques_by_category.items()
        ]) if techniques_by_category else "Nessuna tecnica specificata."

        # Struttura base del filtro
        filters = {
            "AND": {
                "restaurant_name": None,
                "ingredients": [],
                "techniques": [],
                "chef_licenses": [],
                "planet": [],
                "planet_distance": [],
                "exclude_ingredients": [],
                "exclude_techniques": []
            },
            "OR": {
                "ingredients": [],
                "techniques": []
            },
            "min_should_count": 1  # Default per OR
        }

        # User Prompt
        prompt = f"""
        Sei un assistente specializzato in cucina galattica. Il tuo compito è analizzare la richiesta dell'utente e generare una lista di filtri per cercare i piatti più rilevanti.

        ### ISTRUZIONI CRITICHE
        - Estrai SOLO ingredienti, tecniche e altri elementi ESPRESSAMENTE menzionati nella richiesta.
        - NON inventare o inferire informazioni non presenti nel testo originale.
        - OMETTI COMPLETAMENTE le sezioni vuote.
        - Se una sezione principale (AND o OR) non ha nessun elemento, omettila completamente.
        - Se una sottocategoria (ingredients, techniques, ecc.) non ha elementi, omettila completamente.
        - I seguenti sono i nomi dei ristoranti disponibili: {restaurant_names}.
        
        ### TECNICHE DI COTTURA DISPONIBILI DAL MANUALE DI SIRIUS COSMO
        {techniques_str}

        ### REGOLE:
        - Se l'utente chiede che un piatto **contenga più tecniche contemporaneamente** (es. "una tecnica di taglio E una di surgelamento"), le tecniche devono essere incluse in **AND**.
        - Se l'utente chiede **"almeno una"** tecnica da un gruppo (es. "almeno una tecnica di surgelamento o una di taglio"), le tecniche devono essere incluse in **OR** e `min_should_count` deve essere impostato al numero richiesto.
        - Se l'utente specifica un ingrediente da **escludere**, mettilo in `"exclude_ingredients"` nella sezione **AND**.

        **Formato JSON di output:**
        ```json
        {{
            "AND": {{
                "restaurant_name": "nome_ristorante",
                "ingredients": ["ingrediente1", "ingrediente2"],
                "techniques": ["tecnica1", "tecnica2"],
                "chef_licenses": [
                    {{"tipo_licenza": "Temporale", "operator": "==", "grade": 2}},
                    {{"tipo_licenza": "Quantistica", "operator": "<", "grade": 5}}
                ],
                "planet": ["pianeta1", "pianeta2"],  
                "planet_distance": [{{"planet": "pianeta_base", "max_distance": valore}}],  
                "exclude_ingredients": ["ingrediente1", "ingrediente2"],
                "exclude_techniques": ["tecnica1", "tecnica2"]
            }},
            "OR": {{
                "ingredients": ["ingrediente1", "ingrediente2"],  
                "techniques": ["tecnica1", "tecnica2"]  
            }},
            "min_should_count": 2
        }}
        ```

        --- RICHIESTA UTENTE ---
        {user_query}

        --- RISPOSTA (solo JSON) ---
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un assistente specializzato in cucina galattica. Il tuo compito è analizzare la richiesta dell'utente e generare una lista di filtri per cercare i piatti più rilevanti SOLO dalle informazioni esplicitamente menzionate nella richiesta dell'utente. Non inventare o inferire informazioni non presenti nel testo originale. Se un'informazione non è chiaramente specificata, omettila completamente dall'output JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        filters_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()

        # Remove <think>...</think>
        if "<think>" in filters_json and "</think>" in filters_json:
            filters_json = filters_json.split("</think>")[-1].strip()
            
        try:
            filters = json.loads(filters_json)

            # Se il campo techniques non esiste, lo inizializziamo
            if "AND" not in filters:
                filters["AND"] = {}
            if "OR" not in filters:
                filters["OR"] = {}

            # Analizziamo quali tecniche sono richieste e come inserirle nei filtri
            for category, techniques in techniques_by_category.items():
                if any(term in user_query.lower() for term in ["almeno una", "una delle"]):
                    filters["OR"].setdefault("techniques", []).extend(techniques)
                    filters["min_should_count"] = max(len(techniques_by_category), 1)  # Minimo 1 tecnica
                else:
                    filters["AND"].setdefault("techniques", []).extend(techniques)
                    
            print(f"[DEBUG] Filtri generati:\n{json.dumps(filters, indent=4, ensure_ascii=False)}")
            
            return filters

        except json.JSONDecodeError as e:
            print(f"[ERRORE] Errore nel parsing dei filtri JSON: {e}")
            return {}


    def execute(self, user_query):
        """Recupera tecniche di Sirius Cosmo e genera i filtri di ricerca."""
        print("[ToolGenerateFiltersSirius] → Recupero tecniche dal Manuale di Sirius Cosmo.")

        techniques = self.extract_techniques(user_query)

        if not techniques:
            return "Errore: Nessuna tecnica trovata."

        # Generiamo i filtri, includendo le tecniche trovate
        filters = self.generate_filters(user_query, techniques)
        return filters
