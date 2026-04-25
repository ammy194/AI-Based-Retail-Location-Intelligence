import os

def replace_in_file(filepath, old, new):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

files = ["src/feature_engineering.py", "src/demand_model.py", "src/location_model.py"]
replacements = {
    '"population"': '"population_density"',
    'out["population"]': 'out["population_density"]',
    'loc["population"]': 'loc["population_density"]',
    '"income_level"': '"avg_income"',
    'out["income_level"]': 'out["avg_income"]',
    'loc["income_level"]': 'loc["avg_income"]'
}

for file in files:
    for old, new in replacements.items():
        replace_in_file(file, old, new)

print("[OK] Updated column references to population_density and avg_income.")
