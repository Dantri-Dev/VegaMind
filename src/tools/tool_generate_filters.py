import requests
import json
import pandas as pd
import os
import logging
from src.config_loader import ConfigLoader

class ToolGenerateFilters:
    def __init__(self, config_path="config/config.yaml"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        print("Configurazione caricata correttamente.")

        # Recupera le variabili dalla configurazione
        api_key = self.config["google"]["api_key"]
        base_url = self.config["google"]["model"]

        # Costruisci l'URL
        self.url = f"{base_url}{api_key}"

        self.distances_df = pd.read_csv(self.config["paths"]["distances"])
        print("[OK] Distanze planetarie caricate!\n")
    
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

    def generate_filters(self, user_query):
        """Genera filtri dinamici basati sulla richiesta dell'utente."""
        print("\n[STEP] Generazione dei filtri dinamici...\n")
        
        restaurant_names = self.get_restaurant_names(self.config["paths"]["menus_dir"])
        
        # Ottieni i pianeti disponibili dalla matrice delle distanze
        available_planets = list(self.distances_df.columns)[1:]
        
        prompt = f"""
        ### CRITICAL INSTRUCTIONS
        - Extract **ONLY** ingredients, techniques, and other elements **EXPLICITLY** mentioned in the request.
        - **DO NOT** invent or infer information not present in the original text.
        - **COMPLETELY OMIT** empty sections (do not include empty arrays or objects).
        - If a main section (AND/OR) has no elements, omit it entirely.
        - If a subcategory (e.g., ingredients, techniques) has no elements, omit it entirely.
        - The following are the available restaurant names: {restaurant_names}
        - The following are the available planets in the system: {available_planets}

        ### FILTER GENERATION RULES
        1. **AND Conditions**:
        - Use **AND** when the request implies that **all** conditions must be satisfied.
        - Apply to:
            - Chef licenses (the chef must have **all** specified licenses).
            - License grades (if a minimum grade is specified).
            - Excluded ingredients (the dish must **not** contain any of them).
            - Excluded techniques (the dish must **not** contain any of them).
            - Restaurant name (if specified in the request).

        2. **OR Conditions**:
        - Use **OR** when the request implies that **at least one** of the conditions must be true.
        - Apply to:
            - Optional ingredients (e.g., "at least 2 ingredients from X, Y, Z").
            - Use the `min_should_count` field to indicate the minimum number of `should` conditions that must be met.

        3. **Restaurant Name**:
        - Include **ONLY** if the name is explicitly mentioned in the request.
        - Verify against the list: {restaurant_names}.

        4. **Planets**:
        - Include **ONLY** if the planet is explicitly mentioned in the request.
        - Verify against the list: {available_planets}.

        ### CHEF LICENSES
        For chef licenses, specify:
        - The license type (e.g., "Temporal", "Galactic").
        - The comparison operator:
        - "==" for an exact grade.
        - ">=" for a grade greater than or equal to.
        - ">" for a grade strictly greater than.
        - "<=" for a grade less than or equal to.
        - "<" for a grade strictly less than.
        - The required grade.

        ### LICENSE GRADES
        For chef license grades, specify:
        - The comparison operator (e.g., ">=", "==").
        - The minimum required grade (e.g., 3).

        ### PLANETS AND DISTANCES
        For planetary distance references:
        - If the user mentions a specific planet (e.g., "dishes from Tatooine"), include it in the "planet" field in OR.
        - If the user mentions a maximum distance from a planet (e.g., "within X light years from planet Y"), you must:
        1. **NOT** add planets to the "planet" field.
        2. Add **ONLY** the origin planet and the maximum distance in the "planet_distance" field.
        3. The system will automatically calculate which planets are within that distance.

        ### OUTPUT FORMAT
        - Return **ONLY** a valid JSON object.
        - Include **ONLY** sections with valid elements.
        - Example:
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
            "chef_licenses_grades": {{ "operator": ">=", "grade": 3 }},
            "planet": ["pianeta1", "pianeta2"],
            "planet_distance": [{{"planet": "pianeta_base", "max_distance": valore}}]
            "exclude_ingredients": ["ingrediente1", "ingrediente2"],
            "exclude_techniques": ["tecnica1", "tecnica2"]
        }},
        "OR": {{
            "ingredients": ["ingrediente1", "ingrediente2"],  
            "techniques": ["tecnica1", "tecnica2"]  
        }},
        "min_should_count": 2  // Numero minimo di condizioni OR che devono essere soddisfatte
        }}
        ```

        --- USER REQUEST ---
        {user_query}

        --- RESPONSE (JSON only) ---
        """

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_json = response.json()
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            generated_text = generated_text.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(generated_text)
            except json.JSONDecodeError as e:
                print(f"[ERRORE] Errore nel parsing dei filtri JSON: {e}")
                return {}
        else:
            raise Exception(f"Errore nella richiesta API: {response.status_code}, {response.text}")

    def execute(self, user_query):
        """Genera i filtri per la query."""
        print("[ToolGenerateFilters] → Generazione filtri per la query standard.")
        return self.generate_filters(user_query)
