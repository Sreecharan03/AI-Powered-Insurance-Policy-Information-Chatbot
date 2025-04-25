import fitz  # PyMuPDF
import re
import json
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text.strip()



def extract_with_regex(text: str) -> dict:
    def extract_field(field_name: str) -> str:
        # Matches `field_name:` and captures everything until the next label or end of string
        pattern = rf"{field_name}\s*:\s*(.*?)(?=\n(?:\w+\s*:)|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    return {
        "policy_type": extract_field("policy_type"),
        "coverage": extract_field("coverage"),
        "content": extract_field("content")
    }

    
def enrich_chunk_with_zephyr(section_text: str, section_title: str, source: str, pipe) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant helping extract structured information from insurance policy documents. "
                "Your job is to return a valid JSON object with the following fields:\n\n"
                "- section_title: The title of the section (same as input)\n"
                "- content: A cleaned, complete, and meaningful paragraph in natural language summarizing the key information from the section. "
                "This should be plain text — not a dictionary or nested structure. Think like a human explaining this section in full sentences.\n"
                "- policy_type: Extract only if clearly mentioned (e.g., Health, Life, or product name like my:health Suraksha)\n"
                "- coverage: Only if benefits, conditions, or limits are clearly described\n\n"
                "If any field is not present, leave it as an empty string. "
                "Return ONLY a valid JSON object. No extra markdown, explanation, or formatting."
            )
        },
        {
            "role": "user",
            "content": f"Section Title: {section_title}\nContent:\n{section_text}"
        }
    ]

    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
        generated_text = outputs[0]["generated_text"]

        # Extract the JSON portion only
        json_part = generated_text.split("<|assistant|>")[-1].strip()
        #print(json_part)

        try:
            metadata = json.loads(json_part)
            metadata["source"] = source
            metadata.setdefault("section_title", section_title)
            metadata.setdefault("content", section_text)
            metadata["error"] = None

        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed, attempting regex fallback...")
            regex_data = extract_with_regex(section_text)
            #print(regex_data)
            if regex_data["policy_type"] or regex_data["coverage"]:
                metadata = {
                    "source": source,
                    "section_title": section_title,
                    "content": regex_data["content"],
                    "policy_type": regex_data["policy_type"],
                    "coverage": regex_data["coverage"],
                    "error": "LLM parse failed - regex fallback used"
                }
            else:
                print("❌ Both LLM JSON and regex extraction failed.")
                metadata = {
                    "source": source,
                    "section_title": section_title,
                    "content": section_text,
                    "policy_type": "",
                    "coverage": "",
                    "error": "LLM parse failed & regex fallback both failed"
                }


    except Exception as e:
        raise RuntimeError(f"enrich_chunk_with_zephyr failed for section '{section_title}': {str(e)}")
    
    print(metadata)
    return {
        "text": f"Section Title: {metadata['section_title']}\n{metadata['content']}",
        "metadata": metadata
    }


def chunk_text_by_paragraphs(text: str, min_length: int = 100) -> List[Dict]:
    raw_chunks = re.split(r'\n{2,}', text)
    sections = []
    for i, chunk in enumerate(raw_chunks):
        # Remove line breaks and tabs, normalize spacing
        cleaned = re.sub(r'[\n\t\r]+', ' ', chunk)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

        if len(cleaned) >= min_length:
            sections.append({
                "section_title": f"Paragraph {i+1}",
                "content": cleaned
            })
    return sections
