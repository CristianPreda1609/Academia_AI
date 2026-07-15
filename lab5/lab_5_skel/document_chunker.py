# Citirea documentelor cu false sau ceva

import os
import json


def load_n_chunk_docs():
    prompt = ""
    files_to_read = os.listdir("knowledge")
    for file_to_read in files_to_read:
        sub_files = os.listdir(os.path.join("knowledge", file_to_read))
        for sub_file in sub_files:
            if file_to_read == "facts" or file_to_read == "procedures":
                if sub_file.endswith(".json"):
                    with open(os.path.join("knowledge", file_to_read, sub_file), "r", encoding="utf-8") as f:
                        facts = json.load(f)
                        for fact in facts:
                            if not fact.get("always_load"):
                                with open(os.path.join("knowledge", file_to_read, fact.get("id")+'.md'), "r", encoding="utf-8") as f2:
                                    prompt += "\n" + f2.read()