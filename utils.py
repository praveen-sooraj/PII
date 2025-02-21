from model_loader import alpaca_model,gliner_model
import json
import re
import math
from collections import Counter
import zlib
from typing import List, Tuple, Union, Dict
import difflib
import exrex
from pii_config import logger,GENERAL_PATH, SECTOR_DATA_PATH, PII_PATH, VIOLATIONS_PATH, STANDARDS_PATH, CREDENTIAL_PATH, SENSITIVITY_PERSONAL_PATH, CREDENTIAL_PII_PATH




def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            logger.info(f"Successfully loaded JSON data from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while loading {file_path}: {e}")

    return None

sectors_data = load_json(SECTOR_DATA_PATH)

Pii_json = load_json(PII_PATH)

general_list = load_json(GENERAL_PATH)

sensitivity_personal_levels = load_json(SENSITIVITY_PERSONAL_PATH)

Credential_pii_list = load_json(CREDENTIAL_PATH)

Credential_pii_levels = load_json(CREDENTIAL_PII_PATH)

# Load standards and violations from JSON files
def get_standards():
    return load_json(STANDARDS_PATH)

# Load violations from JSON (not fetching via API)
def get_violations_by_standard():
    return load_json(VIOLATIONS_PATH)


def classify_sector_with_alpaca(content):   
    try:
        logger.info("Starting sector classification.")
        prompt = f"Given the following API call content:\n{content}\nWhich sector does this belong to from the following list in one word?"
        
        response = alpaca_model(prompt, max_length=128, do_sample=True)
        
        if not response or "generated_text" not in response[0]:
            logger.error("Unexpected response format from alpaca_model.")
            return None
        
        value = response[0]["generated_text"].strip()
        logger.info(f"Model response: {value}")
        
        sector = find_sector(value)
        logger.info(f"Classified sector: {sector}")
        
        return sector
    
    except Exception as e:
        logger.error(f"Error in classify_sector_with_alpaca: {str(e)}")
        return None

def find_sector(input_text):
    
    try:
        logger.info("Starting sector finding process.")
        input_text = input_text.lower()
        
        for sector_data in sectors_data["sectors"]:
            sector = sector_data["sector"]
            keywords = sector_data["keywords"]
            for keyword in keywords:
                if keyword in input_text:
                    logger.info(f"Match found: {keyword} -> {sector}")
                    return sector
        
        logger.warning("No matching sector found.")
        return input_text
    
    except Exception as e:
        logger.error(f"Error in find_sector: {str(e)}")
        return None
    
def filter_by_label(data, label_type):
    
    try:
        logger.info(f"Filtering data by label: {label_type}")
        for item in data['data']:
            if item['label'] == label_type:
                logger.info(f"Match found for label: {label_type}")
                return item['list']
        
        logger.warning(f"No matching label found: {label_type}")
        return []
    
    except Exception as e:
        logger.error(f"Error in filter_by_label: {str(e)}")
        return []


def filter_by_label(data, label_type):
    
    try:
        logger.info(f"Filtering data by label: {label_type}")
        for item in data['data']:
            if item['label'] == label_type:
                logger.info(f"Match found for label: {label_type}")
                return item['list']
        
        logger.warning(f"No matching label found: {label_type}")
        return []
    
    except Exception as e:
        logger.error(f"Error in filter_by_label: {str(e)}")
        return []

def filter_by_label_sensitivity(data, label_type):
    
    try:
        logger.info(f"Filtering data by label sensitivity: {label_type}")
        for item in data['data']:
            if item['label'] == label_type:
                logger.info(f"Match found for label: {label_type}")
                return item['pii_level']
        
        logger.warning(f"No matching label found: {label_type}")
        return []
    
    except Exception as e:
        logger.error(f"Error in filter_by_label_sensitivity: {str(e)}")
        return []
    
def extract_text_and_keys(data, prefix=""):
    
    text = ""
    key_map = []
    
    try:
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                        sub_text, sub_key_map = extract_text_and_keys(value, f"{prefix}{key}: ")
                        text += sub_text
                        key_map.extend(sub_key_map)
                    except (json.JSONDecodeError, TypeError):
                        text += f"{prefix}{key}: {value} "
                        key_map.append((f"{prefix}{key}", value))
                else:
                    sub_text, sub_key_map = extract_text_and_keys(value, f"{prefix}{key}: ")
                    text += sub_text
                    key_map.extend(sub_key_map)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                sub_text, sub_key_map = extract_text_and_keys(item, f"{prefix}[{idx}]: ")
                text += sub_text
                key_map.extend(sub_key_map)
        
        text = text.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
        return text, key_map
    
    except Exception as e:
        logger.error(f"Error in extract_text_and_keys: {str(e)}")
        return "", []



def extract_text_and_keys(data, prefix=""):
    
    text = ""
    key_map = []
    
    try:
        logger.info("Starting extraction of text and keys.")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                        sub_text, sub_key_map = extract_text_and_keys(value, f"{prefix}{key}: ")
                        text += sub_text
                        key_map.extend(sub_key_map)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Skipping JSON decoding for key: {prefix}{key}")
                        text += f"{prefix}{key}: {value} "
                        key_map.append((f"{prefix}{key}", value))
                else:
                    sub_text, sub_key_map = extract_text_and_keys(value, f"{prefix}{key}: ")
                    text += sub_text
                    key_map.extend(sub_key_map)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                sub_text, sub_key_map = extract_text_and_keys(item, f"{prefix}[{idx}]: ")
                text += sub_text
                key_map.extend(sub_key_map)
        
        text = text.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
        logger.info("Extraction completed successfully.")
        return text, key_map
    
    except Exception as e:
        logger.error(f"Error in extract_text_and_keys: {str(e)}")
        return "", []



def detect_entities_from_dynamic_json(json_data, label):
    logger.info("Starting PII detection process.")
    
    pii_list = filter_by_label(Pii_json, label)
    pii_sensitivity = filter_by_label_sensitivity(Pii_json, label)
    
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input JSON: {str(e)}")
        return []
    
    text, key_map = extract_text_and_keys(data)
    
    if not text.strip():
        logger.warning("No text found in JSON data.")
        return []
    
    key_entity_map = []
    seen_entities = set()
    
    try:
        for pii_type, pii_items in [("General", general_list), ("Credential", Credential_pii_list)]:
            if pii_items:
                entities = gliner_model.predict_entities(text, pii_items, 
                                                  threshold=0.15 if pii_type == "General" else 0.1)
                if entities:
                    for entity in entities:
                        for key, value in key_map:
                            if entity["text"] == value:
                                entity_key = f"{key}-{entity['text']}"
                                if entity_key in seen_entities:
                                    continue
                                seen_entities.add(entity_key)
                                sensitivity_dict = (sensitivity_personal_levels 
                                                    if pii_type == "General" 
                                                    else Credential_pii_levels)
                                sensitivity_level = sensitivity_dict.get(
                                    entity["label"], {}).get('body', 'unknown')
                                
                                key_parts = key.split(': ')
                                param_type = key_parts[0]
                                path = "/".join([part.strip() for part in key_parts[1:]])
                                path = re.sub(r'\[\d+\]', '', path)
                                path = re.sub(r'/{2,}', '/', path)
                                last_key_part = path.split('/')[-1] if '/' in path else key
                                
                                new_data = {
                                    "param_type": param_type,
                                    "path": path.replace(last_key_part, "[ ]"),
                                    "key": last_key_part,
                                    "text": entity["text"],
                                    "label": entity["label"],
                                    "type": pii_type,
                                    "sensitivity_level": sensitivity_level
                                }
                                key_entity_map.append(new_data)
        
        # Process sector-specific PII list
        if pii_list:
            entities = gliner_model.predict_entities(text, pii_list, threshold=0.3)
            if entities:
                for entity in entities:
                    for key, value in key_map:
                        if entity["text"] == value:
                            entity_key = f"{key}-{entity['text']}"
                            if entity_key in seen_entities:
                                continue
                            seen_entities.add(entity_key)
                            
                            sensitivity_level = pii_sensitivity.get(
                                entity["label"], {}).get('body', 'unknown')
                            
                            key_parts = key.split(': ')
                            param_type = key_parts[0]
                            path = "/".join([part.strip() for part in key_parts[1:]])
                            
                            path = re.sub(r'\[\d+\]', '', path)
                            path = re.sub(r'/{2,}', '/', path)
                            
                            last_key_part = path.split('/')[-1] if '/' in path else key
                            
                            new_data = {
                                "param_type": param_type,
                                "path": path.replace(last_key_part, "[ ]"),
                                "key": last_key_part,
                                "text": entity["text"],
                                "label": entity["label"],
                                "type": label,
                                "sensitivity_level": sensitivity_level
                            }
                            key_entity_map.append(new_data)
        
    except Exception as e:
        logger.error(f"Error during entity detection: {str(e)}")
    
    logger.info("PII detection process completed.")
    return key_entity_map

def is_encrypted(text):
    
    # Shannon Entropy calculation
    def shannon_entropy(data):
        try:
            length = len(data)
            if length == 0:
                return 0.0
            freqs = Counter(data)
            entropy = 0.0
            for freq in freqs.values():
                prob = freq / length
                entropy -= prob * math.log2(prob)
            return entropy
        except Exception as e:
            logger.error(f"Error in Shannon entropy calculation: {str(e)}")
            return 0.0


def is_encrypted(text):
    
    # Shannon Entropy calculation
    def shannon_entropy(data):
        try:
            length = len(data)
            if length == 0:
                return 0.0
            freqs = Counter(data)
            entropy = 0.0
            for freq in freqs.values():
                prob = freq / length
                entropy -= prob * math.log2(prob)
            return entropy
        except Exception as e:
            logger.error(f"Error in Shannon entropy calculation: {str(e)}")
            return 0.0
    
    # Compression ratio
    def compression_ratio(data):
        try:
            compressed = zlib.compress(data.encode())
            original_size = len(data.encode())
            compressed_size = len(compressed)
            ratio = compressed_size / original_size if original_size > 0 else 0.0
            return ratio
        except Exception as e:
            logger.error(f"Error in compression ratio calculation: {str(e)}")
            return 0.0
        
def is_encrypted(text):
    
    # Shannon Entropy calculation
    def shannon_entropy(data):
        try:
            length = len(data)
            if length == 0:
                return 0.0
            freqs = Counter(data)
            entropy = 0.0
            for freq in freqs.values():
                prob = freq / length
                entropy -= prob * math.log2(prob)
            return entropy
        except Exception as e:
            logger.error(f"Error in Shannon entropy calculation: {str(e)}")
            return 0.0
    
    # Compression ratio
    def compression_ratio(data):
        try:
            compressed = zlib.compress(data.encode())
            original_size = len(data.encode())
            compressed_size = len(compressed)
            ratio = compressed_size / original_size if original_size > 0 else 0.0
            return ratio
        except Exception as e:
            logger.error(f"Error in compression ratio calculation: {str(e)}")
            return 0.0
    
    # Check for common encryption headers or patterns
    def has_encryption_header(data):
        try:
            common_encryption_patterns = ['$AES$', '$DES$', 'base64']
            return any(pattern in data for pattern in common_encryption_patterns)
        except Exception as e:
            logger.error(f"Error in encryption header check: {str(e)}")
            return False
    
    try:
        # Calculate Shannon Entropy and Compression Ratio
        entropy = shannon_entropy(text)
        compression = compression_ratio(text)
        encryption_flag = has_encryption_header(text)

        # Log calculated values
        logger.info(f"Entropy: {entropy}, Compression Ratio: {compression}, Encryption Header: {encryption_flag}")

        # Thresholds for considering encryption
        if entropy > 4.5 and compression > 1.5 and encryption_flag:
            return True
        return False
    except Exception as e:
        logger.error(f"Error in encryption detection: {str(e)}")
        return False


# Setting up logger


def is_value_masked(value, min_mask_chars=2):
    try:
        # Masking pattern to match '*' and 'X'
        masking_pattern = re.compile(r'[\*X]')
        
        # Find all occurrences of masking characters in the value
        matches = masking_pattern.findall(value)
        logger.debug(f"Value: {value}, Masking Matches: {matches}")
        
        # Check if the number of masking characters is at least the minimum required
        if len(matches) >= min_mask_chars:
            logger.info(f"Value is sufficiently masked. Masking characters: {len(matches)}")
            return True
        logger.info(f"Value is not sufficiently masked. Masking characters: {len(matches)}")
        return False

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False
    

def check_violations_with_input(input_data, standards):
    try:
        result = []
        logger.info("Starting violation check for input data.")

        for entry in input_data:
            try:
                entry_label = entry.get('label', '').lower()  # Convert to lowercase for case-insensitive comparison
                violations_list = []

                # Iterate through the standards to check violations
                for standard in standards:
                    for violating_key in standard.get('violating_keys', []):
                        violating_name = violating_key.get('name', '').lower()

                        # Check if the label matches any violating key name
                        if entry_label == violating_name:
                            text_value = entry.get('text', '')

                            violations_list.append({
                                "standard": standard.get('standard_name', 'Unknown Standard'),
                                "identified": True,
                                "violation": "VALUE NOT MASKED" if not is_value_masked(text_value) else "VALUE IS MASKED."
                            })
                            
                            violations_list.append({
                                "standard": standard.get('standard_name', 'Unknown Standard'),
                                "identified": True,
                                "violation": "VALUE NOT ENCRYPTED" if not is_encrypted(text_value) else "VALUE IS ENCRYPTED."
                            })
                            
                            violations_list.append({
                                "standard": standard.get('standard_name', 'Unknown Standard'),
                                "identified": True,
                                "violation": "NO VALUE" if text_value is None else "VALUE SHOULD NOT BE PRESENT."
                            })

                # Append the violations to the entry
                entry_with_violations = {
                    **entry,
                    "violations": violations_list
                }
                result.append(entry_with_violations)

            except Exception as entry_error:
                logger.exception(f"Error processing entry {entry}: {entry_error}")

        logger.info("Completed violation checks.")
        return result

    except Exception as e:
        logger.exception(f"Unexpected error in check_violations_with_input: {e}")
        return []
    

# Setting up logger


def fetch_standards_and_check_violations(input_data, tenant_id):
    try:
        standards_data = get_standards()
        if not standards_data:
            logger.warning("No standards found.")
            return None

        results = []

        for standard in standards_data:
            try:
                standard_id = standard.get('id')
                standard_name = standard.get('standard_name', 'Unknown Standard')

                logger.info(f"Fetching violations for standard: {standard_name} (ID: {standard_id})")

                violations_data = get_violations_by_standard(standard_id, tenant_id)

                if violations_data:
                    logger.info(f"Checking violations for standard: {standard_name}")

                    # Prepare the standard in the required format for comparison
                    standards = [{
                        "standard_name": standard_name,
                        "violating_keys": [
                            {
                                "name": violating.get('name', ''),
                                "is_masked": violating.get('is_masked', False),
                                "is_encrypted": violating.get('is_encrypted', False),
                                "is_not_presented": violating.get('is_not_presented', False)
                            }
                            for violating in violations_data
                        ]
                    }]

                    result = check_violations_with_input(input_data, standards)
                    if result:
                        results.extend(result)

            except Exception as standard_error:
                logger.exception(f"Error processing standard {standard}: {standard_error}")

        logger.info("Completed checking violations for all standards.")
        return results

    except Exception as e:
        logger.exception(f"Unexpected error in fetch_standards_and_check_violations: {e}")
        return None
    
    
# Setting up logger


def get_object_by_id(array, object_name):
    try:
        logger.debug(f"Searching for object with name: {object_name}")

        for obj in array:
            if obj.get('name') == object_name:
                logger.info(f"Object found: {obj}")
                return obj

        logger.warning(f"Object with name '{object_name}' not found.")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error in get_object_by_id: {e}")
        return None
    

# Setting up logger


def extract_and_group_dynamic_ids(data: dict) -> Dict[str, List[str]]:
    try:
        id_pattern = re.compile(r".*id$", re.IGNORECASE)
        ids_by_group = {}

        logger.debug("Starting extraction of dynamic IDs.")

        for key, value in data.items():
            if id_pattern.match(key):
                if isinstance(value, list):
                    ids_by_group[key] = [str(v) for v in value if v is not None]
                elif value is not None:
                    ids_by_group[key] = [str(value)]
                
                logger.info(f"Extracted IDs for key '{key}': {ids_by_group[key]}")

        logger.debug("Completed extraction of dynamic IDs.")
        return ids_by_group

    except Exception as e:
        logger.exception(f"Unexpected error in extract_and_group_dynamic_ids: {e}")
        return {}
    
    
# Setting up logger


def get_value_pattern(value: str) -> str:
    try:
        logger.debug(f"Analyzing value pattern for: {value}")

        # Handle patterns like 009DEV, 010DEV
        if re.match(r'^\d+[A-Z]+$', value):
            logger.info(f"Pattern identified as 'num_alpha' for value: {value}")
            return 'num_alpha'

        # Handle patterns like 01Emp002Dev
        elif re.match(r'^\d+[A-Za-z]+\d+[A-Za-z]+$', value):
            logger.info(f"Pattern identified as 'mixed_pattern' for value: {value}")
            return 'mixed_pattern'

        # Handle purely numeric sequences
        elif value.isdigit():
            logger.info(f"Pattern identified as 'numeric' for value: {value}")
            return 'numeric'

        # Handle purely alphabetic sequences
        elif value.isalpha():
            logger.info(f"Pattern identified as 'alpha' for value: {value}")
            return 'alpha'

        # Default pattern
        logger.info(f"Pattern identified as 'other' for value: {value}")
        return 'other'

    except Exception as e:
        logger.exception(f"Unexpected error in get_value_pattern for value '{value}': {e}")
        return 'error'
    

# Setting up logger


# Global variables
id_pattern = re.compile(r".*id$", re.IGNORECASE)
ids_by_group = {}

def search_for_ids(obj: Union[dict, list], parent_key: str = ''):
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    search_for_ids(value, key)
                elif id_pattern.match(key) and value is not None:
                    key_pattern = key.replace('_id', '').replace('ID', '')
                    value_pattern = get_value_pattern(str(value))
                    group_key = f"{key_pattern}_{value_pattern}"

                    if group_key not in ids_by_group:
                        ids_by_group[group_key] = []
                    ids_by_group[group_key].append(str(value))

                    logger.info(f"Grouped ID: {value} under key {group_key}")

        elif isinstance(obj, list):
            for item in obj:
                search_for_ids(item, parent_key)

    except Exception as e:
        logger.exception(f"Unexpected error in search_for_ids: {e}")

def extract_and_group_ids(data: dict) -> Dict[str, List[str]]:
    try:
        global ids_by_group
        ids_by_group = {}  # Reset global storage for fresh run

        output_data = data.get("output", {})
        logger.debug("Starting ID extraction process.")
        search_for_ids(output_data)

        logger.info("Completed ID extraction process.")
        return ids_by_group

    except Exception as e:
        logger.exception(f"Unexpected error in extract_and_group_ids: {e}")
        return {}
    
    



def increment_mixed_sequence(value: str) -> str:
    try:
        logger.debug(f"Attempting to increment sequence for value: {value}")

        # Handle patterns like 009DEV, 010DEV
        num_alpha_match = re.match(r'^(\d+)([A-Z]+)$', value)
        if num_alpha_match:
            num, alpha = num_alpha_match.groups()
            next_num = str(int(num) + 1).zfill(len(num))
            incremented_value = f"{next_num}{alpha}"
            logger.info(f"Incremented '{value}' to '{incremented_value}' (num_alpha pattern)")
            return incremented_value

        # Handle patterns like 01Emp002Dev
        mixed_match = re.match(r'^(\d+)([A-Za-z]+)(\d+)([A-Za-z]+)$', value)
        if mixed_match:
            prefix_num, prefix_alpha, suffix_num, suffix_alpha = mixed_match.groups()
            next_suffix_num = str(int(suffix_num) + 1).zfill(len(suffix_num))
            incremented_value = f"{prefix_num}{prefix_alpha}{next_suffix_num}{suffix_alpha}"
            logger.info(f"Incremented '{value}' to '{incremented_value}' (mixed_pattern)")
            return incremented_value

        logger.warning(f"No matching pattern found for value: {value}. Returning original value.")
        return value

    except Exception as e:
        logger.exception(f"Unexpected error in increment_mixed_sequence for value '{value}': {e}")
        return value



def sequence_similarity_analysis(ids: List[str]) -> Tuple[bool, List[float], str]:
    try:
        if len(ids) <= 1:
            logger.warning("Only one or no ID provided, returning default values.")
            return False, [], "single"
        
        similarity_scores = []
        consistent = True

        # Sort IDs to ensure proper sequence checking
        sorted_ids = sorted(ids)
        logger.debug(f"Sorted IDs for analysis: {sorted_ids}")

        # Determine sequence type from first value
        first_value = sorted_ids[0]
        if re.match(r'^\d+[A-Z]+$', first_value):
            sequence_type = "num_alpha"
        elif re.match(r'^\d+[A-Za-z]+\d+[A-Za-z]+$', first_value):
            sequence_type = "mixed_pattern"
        elif first_value.isdigit():
            sequence_type = "numeric"
        elif first_value.isalpha():
            sequence_type = "alpha"
        else:
            sequence_type = "other"
        
        logger.info(f"Determined sequence type: {sequence_type}")

        for i in range(len(sorted_ids) - 1):
            score = difflib.SequenceMatcher(None, sorted_ids[i], sorted_ids[i + 1]).ratio()
            similarity_scores.append(score)
            logger.debug(f"Similarity score between '{sorted_ids[i]}' and '{sorted_ids[i + 1]}': {score}")

            if score < 0.45:
                consistent = False

        logger.info(f"Sequence consistency: {consistent}")
        return consistent, similarity_scores, sequence_type

    except Exception as e:
        logger.exception(f"Unexpected error in sequence_similarity_analysis: {e}")
        return False, [], "error"
    



def analyze_dynamic_ids(body: dict):
    try:
        logger.debug("Starting dynamic ID analysis.")
        grouped_ids = extract_and_group_dynamic_ids(body)
        results = {}

        for group_key, ids in grouped_ids.items():
            logger.info(f"Analyzing group: {group_key} with IDs: {ids}")

            consistent, similarity_scores, sequence_type = sequence_similarity_analysis(ids)

            if consistent:
                # Handle different sequence types
                if sequence_type in ["num_alpha", "mixed_pattern"]:
                    next_value = increment_mixed_sequence(sorted(ids)[-1])
                else:
                    next_value = process_last_value(ids)

                results[group_key] = {
                    "consistent": True,
                    "sequence_type": sequence_type,
                    "similarity_scores": similarity_scores,
                    "next_value": next_value,
                    "pattern": None,
                    "random_value": None
                }

                logger.info(f"Consistent sequence detected for {group_key}. Next value: {next_value}")
            
            else:
                pattern = generate_detailed_pattern(ids[0])
                random_value = exrex.getone(pattern)

                results[group_key] = {
                    "consistent": False,
                    "sequence_type": sequence_type,
                    "similarity_scores": similarity_scores,
                    "pattern": pattern,
                    "next_value": None,
                    "random_value": random_value
                }

                logger.warning(f"Inconsistent sequence detected for {group_key}. Pattern: {pattern}, Random value: {random_value}")

        logger.debug("Completed dynamic ID analysis.")
        return results

    except Exception as e:
        logger.exception(f"Unexpected error in analyze_dynamic_ids: {e}")
        return {}
    
    


def process_last_value(lst: List[str]) -> str:
    try:
        if not lst:
            logger.warning("Empty list provided to process_last_value, returning None.")
            return None

        sorted_list = sorted(lst)
        last_value = sorted_list[-1]

        logger.info(f"Processing last value: {last_value}")

        if last_value.isdigit():
            next_value = str(int(last_value) + 1).zfill(len(last_value))
            logger.info(f"Processed numeric sequence: {next_value}")
            return next_value

        # Split into prefix and number
        match = re.match(r"([A-Za-z]+)(\d+)", last_value)
        if match:
            prefix, number = match.groups()
            next_number = str(int(number) + 1).zfill(len(number))
            next_value = f"{prefix}{next_number}"
            logger.info(f"Processed alphanumeric sequence: {next_value}")
            return next_value

        # If no numeric pattern is detected, try alphabetical increment
        next_value = increment_alphabetical_sequence(last_value)
        logger.info(f"Processed alphabetical sequence: {next_value}")
        return next_value

    except Exception as e:
        logger.exception(f"Unexpected error in process_last_value: {e}")
        return None
    
    


def generate_detailed_pattern(id_text: str) -> str:
    try:
        logger.debug(f"Generating detailed pattern for: {id_text}")

        pattern = ""
        current_type = None
        count = 0

        for char in id_text:
            if char.isdigit():
                char_type = "digit"
            elif char.isalpha():
                char_type = "alpha"
            else:
                char_type = "other"

            if char_type == current_type:
                count += 1
            else:
                if current_type:
                    pattern += f"{current_type}{{{count}}}"
                current_type = char_type
                count = 1

        if current_type:
            pattern += f"{current_type}{{{count}}}"

        logger.info(f"Generated pattern: {pattern}")
        return pattern

    except Exception as e:
        logger.exception(f"Unexpected error in generate_detailed_pattern: {e}")
        return ""
    
    


def generate_detailed_pattern(id_text: str) -> str:
    try:
        logger.debug(f"Generating detailed pattern for: {id_text}")

        def add_to_pattern(type_char, count):
            if type_char == "upper":
                return f"[A-Z]{{{count}}}"
            elif type_char == "lower":
                return f"[a-z]{{{count}}}"
            elif type_char == "digit":
                return f"\\d{{{count}}}"
            elif type_char == "special":
                return re.escape(char) * count
            return ""

        pattern = ""
        current_type = None
        count = 0

        for char in id_text:
            if char.isupper():
                char_type = "upper"
            elif char.islower():
                char_type = "lower"
            elif char.isdigit():
                char_type = "digit"
            else:
                char_type = "special"

            if char_type == current_type:
                count += 1
            else:
                if current_type:
                    pattern += add_to_pattern(current_type, count)
                current_type = char_type
                count = 1

        if current_type:
            pattern += add_to_pattern(current_type, count)

        logger.info(f"Generated pattern: {pattern}")
        return pattern

    except Exception as e:
        logger.exception(f"Unexpected error in generate_detailed_pattern: {e}")
        return ""
    
    


def increment_alphabetical_sequence(seq: str) -> str:
    """Increment an alphabetical sequence like AAA to AAB."""
    try:
        logger.debug(f"Incrementing sequence: {seq}")

        if seq.isalpha():
            # Convert sequence to uppercase letters list
            seq_list = list(seq.upper())
            i = len(seq_list) - 1
            
            while i >= 0:
                if seq_list[i] == 'Z':  # Reset current character to 'A' and carry to the next
                    seq_list[i] = 'A'
                    i -= 1
                else:
                    seq_list[i] = chr(ord(seq_list[i]) + 1)
                    break
            else:
                seq_list.insert(0, 'A')  # If all characters were 'Z', add a new 'A' at the beginning
            
            new_seq = ''.join(seq_list)
            logger.info(f"Incremented sequence: {new_seq}")
            return new_seq

        logger.warning(f"Invalid sequence provided: {seq}, returning unchanged.")
        return seq

    except Exception as e:
        logger.exception(f"Unexpected error in increment_alphabetical_sequence: {e}")
        return seq