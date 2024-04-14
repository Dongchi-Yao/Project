# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:22:25 2022

@author: smrya
"""

import re

def split_into_sentences(text):
    # Regular expression patterns to identify special cases in text
    alphabets = "([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov)"

    # Preprocessing the text to deal with known edge cases
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, r"\1<prd>", text)
    text = re.sub(websites, r"<prd>\1", text)
    text = re.sub(r"\s" + alphabets + r"[.] ", r" \1<prd> ", text)
    text = re.sub(acronyms + r" " + starters, r"\1<stop> \2", text)
    text = re.sub(alphabets + r"[.]" + alphabets + r"[.]" + alphabets + r"[.]", r"\1<prd>\2<prd>\3<prd>", text)
    text = re.sub(alphabets + r"[.]" + alphabets + r"[.]", r"\1<prd>\2<prd>", text)
    text = re.sub(r" " + suffixes + r"[.] " + starters, r" \1<stop> \2", text)
    text = re.sub(r" " + suffixes + r"[.]", r" \1<prd>", text)
    text = re.sub(r" " + alphabets + r"[.]", r" \1<prd>", text)

    # Handling special punctuation within quotes
    text = re.sub(r'\." ', r'". ', text)
    text = re.sub(r'\?" ', r'"? ', text)
    text = re.sub(r'!" ', r'"! ', text)

    # Replace placeholder markers with actual punctuation and stops
    text = text.replace("<prd>", ".")
    text = text.replace(". ", ".<stop>")
    text = text.replace("? ", "?<stop>")
    text = text.replace("! ", "!<stop>")

    # Split the text into sentences using the <stop> marker
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences
