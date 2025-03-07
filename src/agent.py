import groq
import json
import pandas as pd
from typing import List, Dict, Any
from src.config_loader import ConfigLoader
import requests
import logging
from qdrant_client.http import models
import time

class VegaMindAgent:
    def __init__(self, tools, qdrant_handler, config_path="config/config.yaml"):

        # load tools
        self.tools = tools
        
        # Configura il logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print("\n[INIT] Caricamento della configurazione...\n")
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        print("[OK] Configurazione caricata correttamente!\n")

        self.qdrant_handler = qdrant_handler

        print("[INIT] Inizializzazione del client Groq...\n")
        self.client = groq.Client(api_key=self.config["groq"]["api_key"])
        self.model = self.config["groq"]["model"]
        print("[OK] Client Groq inizializzato con il modello:", self.model, "\n")

        print("[INIT] Caricamento delle informazioni sui piatti e distanze...\n")
        with open(self.config["paths"]["dish_mapping"], 'r') as f:
            self.dish_mapping = json.load(f)

        self.distances_df = pd.read_csv(self.config["paths"]["distances"], index_col=0)
        print("[OK] Mappatura piatti e distanze planetarie caricate!\n")
        
    def decide_tool(self, user_query):
        """Usa il modello DeepSeek per determinare quale tool chiamare."""

        prompt = f"""
        ### CONTESTO:
        - Il database contiene informazioni su piatti galattici, ingredienti, tecniche di cottura, pianeti, licenze e altro.
        - Il Manuale di Sirius Cosmo è un importante riferimento che contiene tecniche speciali.

        ### ISTRUZIONI:
        1. Scegli "generate_filters_sirius" quando:
        - La domanda menziona esplicitamente "Sirius Cosmo", "manuale di Sirius", o "di Sirius Cosmo"
        - La domanda chiede quali sono i piatti preparati utilizzando sia tecniche di impasto che tecniche di taglio
        
        2. Scegli "generate_filters" per tutte le altre domande, incluse quelle che:
        - Chiedono di piatti con specifici ingredienti
        - Riguardano licenze o pianeti senza riferimento al manuale di Sirius Cosmo
        - Contengono richieste di filtraggio complesse (es. "almeno 2 tra...")
        - Esclusioni di piatti

        ### ESEMPI:
        - "Quali piatti contengono Erba Pipa?" → generate_filters
        - "Quali piatti usano tecniche di Sirius Cosmo?" → generate_filters_sirius
        - "Quali piatti sono preparati su Asgard?" → generate_filters
        - Se non richiesta non fa riferimento a generate_filters o generate_filters_sirius allora rispondi → none

        **Esempi di output:**
        ```json
        {{
            "tool": "generate_filters"
        }}
        ```
        ```json
        {{
            "tool": "generate_filters_sirius"
        }}
        ```
        ```json
        {{
            "tool": "none"
        }}
        ```

        --- DOMANDA UTENTE ---
        {user_query}

        --- RISPOSTA (solo JSON) ---
        """

        # Chiamata al LLM per decidere il tool da usare
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un assistente AI specializzato in cucina galattica. Il tuo compito è analizzare la domanda dell'utente e determinare quale tool utilizzare per interrogare correttamente il database."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        response_content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        
        # Remove <think>...</think>
        if "<think>" in response_content and "</think>" in response_content:
            response_content = response_content.split("</think>")[-1].strip()

        # Verifica se la risposta è vuota o non è un JSON valido
        if not response_content or not response_content.startswith("{"):
            print("Errore: La risposta del modello non è un JSON valido.")
            print("Risposta:", json.loads(response_content))
            # Assegna un valore di default
            tool_selected = {"tool": "generate_filters"}
        else:
            try:
                tool_selected = json.loads(response_content)
                print(f"\nRisposta ricevuta: {json.loads(response_content)}\n")
            except json.JSONDecodeError as e:
                print("Errore nel parsing del JSON:", e)
                print(f"\nRisposta ricevuta: {response_content}\n")
                # Assegna un valore di default
                tool_selected = {"tool": "generate_filters"}

        print(f"[Agent Principale] → Tool selezionato: {tool_selected['tool']}\n")
        return tool_selected["tool"]

    def retrieve_relevant_context(self, filters, k=None):
        """Recupera i nomi dei piatti rilevanti dalla knowledge base utilizzando solo i filtri."""

        if k is None:
            k = self.config["agent"]["top_k_results"]

        # Costruisci il filtro per Qdrant
        qdrant_filter = None
        if filters:
            qdrant_filter = self.build_qdrant_filter(filters)
            print(f"[STEP] Filtro Qdrant costruito: {qdrant_filter}\n")

        print(f"[STEP] Ricerca nei documenti più rilevanti in Qdrant (Top {k} risultati)...\n")
        
        try:
            # Esegui una ricerca basata solo sui filtri
            search_result = self.qdrant_handler.search_with_filters(qdrant_filter, k)
            print(f"[OK] {len(search_result)} documenti trovati!\n")
        except Exception as e:
            print(f"[ERRORE] Errore durante la ricerca con filtri: {e}")
            return "", ""  # Restituisci due stringhe vuote in caso di errore

       # Estrai i nomi dei piatti, ingredienti e tecniche dai risultati
        dish_names = set()  # Usiamo un set per evitare duplicati
        ingredients = set()  # Set per evitare duplicati negli ingredienti
        techniques = set()  # Set per evitare duplicati nelle tecniche

        for result in search_result:
            if "dish" in result.payload:
                dish_names.add(result.payload["dish"])
            if "ingredients" in result.payload:
                ingredients.update(result.payload["ingredients"])  # Aggiungi gli ingredienti al set
            if "techniques" in result.payload:
                techniques.update(result.payload["techniques"])  # Aggiungi le tecniche al set

        # Converti i set in stringhe separate da virgole
        dish_names_str = ", ".join(dish_names)
        ingredients_str = ", ".join(ingredients)  # Unisci gli ingredienti in una stringa separata da virgole
        techniques_str = ", ".join(techniques)  # Unisci le tecniche in una stringa separata da virgole

        print(f"[STEP] Nomi dei piatti estratti: {dish_names_str}\n")
        print(f"[STEP] Ingredienti estratti: {ingredients_str}\n")
        print(f"[STEP] Tecniche estratte: {techniques_str}\n")

        # Normalizza i nomi dei piatti
        dishes = [dish.strip() for dish in dish_names_str.split(',') if dish.strip()]

        return dishes, ingredients_str, techniques_str


    
    def build_qdrant_filter(self, filters):
        """Costruisce un filtro Qdrant basato sui filtri generati dal modello LLM, gestendo sia AND che OR."""
        
        # Se filters è una stringa JSON, convertila in un dizionario
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)  # Converte la stringa JSON in un dizionario
            except json.JSONDecodeError as e:
                print("Errore nel parsing del JSON:", e)
                return None

        # Verifica che filters sia un dizionario
        if not isinstance(filters, dict):
            print("Errore: filters non è un dizionario.")
            return None

        def normalize_value(value):
            """Normalizza i valori in lowercase."""
            if isinstance(value, str):
                return value.lower()
            elif isinstance(value, list):
                return [v.lower() if isinstance(v, str) else v for v in value]
            return value

        # Estrai min_should_count dal dizionario filters, se presente
        min_should_count = 1  # Valore predefinito
        if "min_should_count" in filters:
            min_should_count = filters.pop("min_should_count")  # Rimuoviamo questa chiave per non confonderla con i filtri

        # Normalizza solo i valori di 'ingredients' e 'techniques'
        normalized_filters = {}
        for operator, conditions in filters.items():
            normalized_conditions = {}
            for key, values in conditions.items():
                if key in ["ingredients", "techniques"]: # Applica la normalizzazione solo a questi campi
                    normalized_conditions[key] = normalize_value(values)
                else:
                    normalized_conditions[key] = values  # Mantieni gli altri campi invariati
            normalized_filters[operator] = normalized_conditions

        must_conditions = []  # Condizioni che devono essere soddisfatte (AND)
        should_conditions = []  # Almeno una di queste deve essere soddisfatta (OR)
        must_not_conditions = []  # Condizioni da escludere
        
        # Definizione delle range_conditions
        range_conditions = {
            "==": lambda value: models.MatchValue(value=value),  # Restituisce un'istanza di MatchValue
            ">=": lambda value: models.Range(gte=value),         # Restituisce un'istanza di Range
            ">": lambda value: models.Range(gt=value),           # Restituisce un'istanza di Range
            "<=": lambda value: models.Range(lte=value),         # Restituisce un'istanza di Range
            "<": lambda value: models.Range(lt=value)            # Restituisce un'istanza di Range
        }

        # **Filtri AND (condizioni obbligatorie)**
        and_filters = normalized_filters.get("AND", {})

        # **Licenze dello chef**
        if "chef_licenses" in and_filters:
            for license_info in and_filters["chef_licenses"]:
                license_type = license_info.get("tipo_licenza")
                operator = license_info.get("operator")
                grade = license_info.get("grade")

                if license_type and operator and grade is not None:
                    key = f"chef_license_{license_type}"
                    if operator in range_conditions:
                        # Ottieni l'istanza corretta di Range o MatchValue
                        condition = range_conditions[operator](grade)
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                **{"match" if operator == "==" else "range": condition}
                            )
                        )
                    else:
                        self.logger.warning(f"Operatore non supportato: {operator} per la licenza {license_type}")

        if "chef_licenses_grades" in and_filters:
            grades_info = and_filters["chef_licenses_grades"]
            operator = grades_info.get("operator")
            grade = grades_info.get("grade")

            if operator and grade is not None:
                if operator in range_conditions:
                    # Aggiungi la condizione principale (must)
                    if operator == "==":
                        condition = models.MatchAny(any=[grade])
                    else:
                        condition = range_conditions[operator](grade)
                    must_conditions.append(
                        models.FieldCondition(
                            key="chef_licenses_grades",
                            **{"match" if operator == "==" else "range": condition}
                        )
                    )

                    # Aggiungi la condizione must_not per escludere i documenti con almeno un valore che non soddisfa la condizione
                    if operator == ">=":
                        must_not_conditions.append(
                            models.FieldCondition(
                                key="chef_licenses_grades",
                                range=models.Range(lt=grade)
                            )
                        )
                    elif operator == ">":
                        must_not_conditions.append(
                            models.FieldCondition(
                                key="chef_licenses_grades",
                                range=models.Range(lte=grade)
                            )
                        )
                    elif operator == "<=":
                        must_not_conditions.append(
                            models.FieldCondition(
                                key="chef_licenses_grades",
                                range=models.Range(gt=grade)
                            )
                        )
                    elif operator == "<":
                        must_not_conditions.append(
                            models.FieldCondition(
                                key="chef_licenses_grades",
                                range=models.Range(gte=grade)
                            )
                        )
                    elif operator == "==":
                        # Per "==", non è necessario aggiungere una condizione must_not
                        pass
                else:
                    self.logger.warning(f"Operatore non supportato: {operator} per chef_licenses_grades")
            
        # **Nome del ristorante**
        if "restaurant_name" in and_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="restaurant_name",
                    match=models.MatchValue(value=and_filters["restaurant_name"])
                )
            )
            
        # **Pianeti menzionati**
        if "planet" in and_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="planet",
                    match=models.MatchAny(any=and_filters["planet"])
                )
            )

        # **Filtri per la distanza planetaria**
        all_planets_within_range = []
        for planet_info in and_filters.get("planet_distance", []):
            planet = planet_info.get("planet")
            max_distance = planet_info.get("max_distance")
            if planet and max_distance:
                all_planets_within_range.extend(self.get_planets_within_distance(planet, max_distance))

        if all_planets_within_range:
            must_conditions.append(
                models.FieldCondition(
                    key="planet",
                    match=models.MatchAny(any=list(set(all_planets_within_range)))
                )
            )

        # **Ingredienti (AND)**
        for ingredient in and_filters.get("ingredients", []):
            must_conditions.append(
                models.FieldCondition(
                    key="ingredients",
                    match=models.MatchValue(value=ingredient)
                )
            )

        # **Ingredienti da escludere**
        for ingredient in and_filters.get("exclude_ingredients", []):
            must_not_conditions.append(
                models.FieldCondition(
                    key="ingredients",
                    match=models.MatchValue(value=ingredient)
                )
            )

        # **Tecniche (AND)**
        for technique in and_filters.get("techniques", []):
            must_conditions.append(
                models.FieldCondition(
                    key="techniques",
                    match=models.MatchValue(value=technique)
                )
            )

        # **Tecniche da escludere**
        for technique in and_filters.get("exclude_techniques", []):
            must_not_conditions.append(
                models.FieldCondition(
                    key="techniques",
                    match=models.MatchValue(value=technique)
                )
            )

        # **Filtri OR (almeno una condizione deve essere soddisfatta)**
        or_filters = normalized_filters.get("OR", {})

        for ingredient in or_filters.get("ingredients", []):
            should_conditions.append(
                models.FieldCondition(
                    key="ingredients",
                    match=models.MatchValue(value=ingredient)
                )
            )

        for technique in or_filters.get("techniques", []):
            should_conditions.append(
                models.FieldCondition(
                    key="techniques",
                    match=models.MatchValue(value=technique)
                )
            )


        # **Costruisci il filtro Qdrant**
        qdrant_filter = models.Filter(
            must=must_conditions or None,
            should=should_conditions or None,
            must_not=must_not_conditions or None,
            min_should=models.MinShould(conditions=should_conditions, min_count=min_should_count) if should_conditions else None
        )
        
        return qdrant_filter

    
    def get_planets_within_distance(self, planet, max_distance):
        """Restituisce i pianeti entro una certa distanza da un pianeta specifico."""
        if planet not in self.distances_df.columns:
            print(f"[ERRORE] Pianeta '{planet}' non trovato nel file delle distanze.")
            return []
        
        # Seleziona i pianeti entro la distanza data
        planets_within_range = self.distances_df.loc[self.distances_df[planet] <= max_distance].index.tolist()

        # Restituisci i nomi dei pianeti
        return planets_within_range

    def get_dish_ids(self, dish_names):
        """Converte i nomi dei piatti in ID."""
        print("\n[STEP] Conversione dei nomi dei piatti in ID...\n")
        dish_ids = [str(self.dish_mapping[name]) for name in dish_names if name in self.dish_mapping]
        print("[OK] ID piatti trovati:", dish_ids, "\n")
        return dish_ids

    def process_query(self, row_id, query, chat=False):
        """Elabora una query e restituisce il risultato finale."""
        
        time.sleep(10)
        
        print(f"\n[PROCESS] Inizio elaborazione della query: {query}\n")
        
        # Chiediamo al LLM quale tool usare
        selected_tool = self.decide_tool(query)
        print("selected_tool", selected_tool)

        # Verifica se il tool selezionato è None
        if not selected_tool or selected_tool == 'none':
            logging.debug(f"Selected tool is None. Proceeding with empty parameters.")
            
            # Se il tool selezionato è None, gestisci il caso senza filtro
            if chat:
                response = self.get_dish_response(query, dishes="", ingredients_str="", techniques_str="")
                logging.debug(f"Response from get_dish_response: {response}")
                return {
                    "success": True,
                    "result": response
                }
            
            logging.debug(f"No filters found for the request.")
            return {
                "success": False,
                "result": "Nessun filtro trovato per la tua richiesta."
            }

        # Se il tool è stato selezionato, esegui il tool
        logging.debug(f"Selected tool: {selected_tool}")
        filters = self.tools[selected_tool].execute(query)
        logging.debug(f"Filtri generati: {filters}")

        # Se i filtri sono vuoti, restituisci un messaggio di errore
        if not filters:
            logging.debug(f"No filters generated, returning error.")
            return {
                "success": False,
                "result": "Nessun filtro generato per la tua richiesta."
            }

        # Recupero contesto basato sui filtri
        dish_names, ingredients_str, techniques_str = self.retrieve_relevant_context(filters)
        logging.debug(f"Dish names: {dish_names}, Ingredients: {ingredients_str}, Techniques: {techniques_str}")

        # Logica per la chat
        if chat:
            if dish_names:
                response = self.get_dish_response(query, dish_names, ingredients_str, techniques_str)
                logging.debug(f"Response with dishes found: {response}")
            else:
                response = "Mi dispiace, non ho trovato piatti correlati alla tua richiesta."
                logging.debug(f"No dishes found, response: {response}")
            
            return {
                "success": True,
                "result": response
            }

        # Se non è chat, trattiamo la conversione in ID
        dish_ids = self.get_dish_ids(dish_names)
        logging.debug(f"Dish IDs: {dish_ids}")

        if not dish_ids:
            logging.debug(f"No dish IDs found, returning error.")
            return {
                "success": False,
                "result": "Nessun piatto trovato."
            }

        logging.debug(f"[PROCESS] Elaborazione completata! Risultato finale: {dish_ids}")

        return {
            "row_id": row_id,
            "result": ",".join(dish_ids)
        }
        
    def get_dish_response(self, query, dishes, ingredients_str, techniques_str):
        """Genera una risposta confermando la richiesta dell'utente sui piatti cercati, includendo descrizioni, ingredienti e tecniche."""

        # Se c'è un solo piatto, struttura la risposta al singolare
        if isinstance(dishes, str):
            # Costruzione della risposta per un singolo piatto
            response_text = f"Certo! Ecco il piatto che cercavi: {dishes}."
            if ingredients_str:
                response_text += f" Ingredienti: {ingredients_str}."
            if techniques_str:
                response_text += f" Tecniche: {techniques_str}."

        # Se ci sono più piatti, struttura la risposta al plurale
        elif isinstance(dishes, list):
            dish_details = []
            for dish in dishes:
                details = f"{dish}"
                # Aggiungi ingredienti e tecniche per ogni piatto
                if ingredients_str:
                    details += f" Ingredienti: {ingredients_str}."
                if techniques_str:
                    details += f" Tecniche: {techniques_str}."
                dish_details.append(details)
            response_text = f"Ecco qui i piatti che mi hai chiesto: {', '.join(dish_details)}."

        # Se non c'è nessun piatto trovato
        if not dishes:
            response_text = "Mi dispiace, non ho trovato esattamente il piatto che cerchi. Forse intendevi qualcosa di simile? Puoi riformulare la richiesta?"

        # Prompt
        prompt = f"""
        Sei un assistente esperto in cucina. Il tuo compito è confermare la richiesta dell'utente riguardo ai piatti culinari e fornire una risposta chiara ed efficace.

        ### ISTRUZIONI:
        - Se l'utente ha chiesto un solo piatto e lo riconosci, conferma la richiesta in modo naturale, ad esempio: "{response_text}"
        - Se l'utente ha chiesto più piatti, elencali in modo fluido e naturale, includendo ingredienti e tecniche quando disponibili.
        - Se non riconosci il piatto, riformula la richiesta per cercare di chiarire cosa sta cercando l'utente.
        - Non rispondere a domande che non siano legate al campo culinario. Se l'utente pone una domanda fuori tema, rifiuta educatamente e informa che il tuo campo di competenza è solo la cucina.
        - Mantieni un tono professionale ma amichevole.

        ---
        **Utente chiede:** {query}
        **Risposta:** "{', '.join(dishes)}"
        """


        # Chiamata al modello per generare la risposta
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        # Estrai la risposta dal modello
        response_content = response.choices[0].message.content.strip()
        
        # Rimuovi eventuali tag <think>...</think>
        if "<think>" in response_content and "</think>" in response_content:
            response_content = response_content.split("</think>")[-1].strip()

        return response_content

    # Mi sarebbe piaciuto effettuare un check intermedio per validare la risposta.
    # Una sorta di "2 Step Verification", ma non ci sono arrivato con i tempi e ho inserito tutte le istruzioni nel prompt principale.
    
    # def validate_dish_selection(self, dish_names, query, rules_context):
    #     """Verifica la validità dei piatti selezionati rispetto alle regole del Codice Galattico."""
    #     print("\n[STEP] Inizio verifica regole Codice Galattico...\n")
    #     validation_prompt = f"""
    #     DOMANDA: {query}
        
    #     REGOLE DEL CODICE GALATTICO:
    #     {rules_context}
        
    #     PIATTI SELEZIONATI:
    #     {', '.join(dish_names)}
        
    #     Verifica se questi piatti rispettano tutte le regole del Codice Galattico. 
    #     Rimuovi quelli non validi. Restituisci SOLO i nomi dei piatti validi separati da virgola.
    #     """

    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=[{"role": "user", "content": validation_prompt}]
    #     )
    #     validated_dishes = response.choices[0].message.content.strip().split(',')
    #     validated_dishes = [dish.strip() for dish in validated_dishes if dish.strip()]
    #     print("[OK] Piatti validati:", validated_dishes, "\n")

    #     return validated_dishes
