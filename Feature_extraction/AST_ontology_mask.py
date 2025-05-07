# =============================================================================
# File: AST_ontology.py
# Description: This script generates binary masks for specific audio related features from AudioSet.
# Author: Vojtěch Nekl
# Created: 28.04.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================

import json
import numpy as np
from transformers import ASTForAudioClassification

class OntologyMaskGenerator:
    """
    Generates binary masks for audio classification filtering based on categories.
    
    Used to filter out AST non-music classes
    """
    def __init__(self, ontology_path, model_name):
        self.model_clf = ASTForAudioClassification.from_pretrained(model_name)
        self.config = self.model_clf.config
        self.labels = self.config.id2label

        # Load the ontology JSON file
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)

        self.id_to_node = {node['id']: node for node in self.ontology}

    def find_category_by_name(self, name):
        for node in self.ontology:
            if node['name'].lower() == name.lower():
                return node
        return None

    def get_all_descendants(self, category):
        descendants = set()

        def collect_children(node):
            descendants.add((node['id'], node['name']))
            for child_id in node.get('child_ids', []):
                if child_id in self.id_to_node:
                    collect_children(self.id_to_node[child_id])

        collect_children(category)
        return descendants

    def generate_masks(self, categories):
        masks = {}
        for category_name in categories:
            category = self.find_category_by_name(category_name)
            if category:
                subtree = self.get_all_descendants(category)
                binary_mask = np.zeros(len(self.labels), dtype=int)
                desired_class_names = [name for _, name in subtree]
                desired_indices = [idx for idx, label in self.labels.items() if label in desired_class_names]

                for index in desired_indices:
                    binary_mask[index] = 1

                masks[category_name] = binary_mask
            else:
                print(f"Category '{category_name}' not found in the ontology.")
                masks[category_name] = None

        return masks
    
    # function to convert binary mask into list of category names
    def get_category_names(self, mask):
        category_names = []
        for idx, value in enumerate(mask):
            if value == 1:
                category_names.append(self.labels[idx])
        return category_names