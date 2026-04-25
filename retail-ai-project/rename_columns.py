import os

def replace_in_file(filepath, old, new):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

files = ["src/feature_engineering.py", "src/demand_model.py", "src/location_model.py"]
replacements = {
    "avg_income": "income_level",
    "num_competitors": "competitors",
    '"accessibility"': '"accessibility_score"',
    'out["accessibility"]': 'out["accessibility_score"]',
    '"parking"': '"parking_availability"',
    'out["parking"]': 'out["parking_availability"]'
}

for file in files:
    for old, new in replacements.items():
        replace_in_file(file, old, new)

print("[OK] Updated column references in existing files.")
