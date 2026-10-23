
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import Crippen


# SMILES клотримазола
smiles = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"

# Создаём молекулу
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("Не удалось распознать SMILES клотримазола.")

# Рассчитываем молекулярные дескрипторы
descriptors = {
    "Drug_Molecular_Weight": Descriptors.MolWt(mol),
    "Drug_MolLogP_RDKit": Crippen.MolLogP(mol),
    "Drug_HBD": Lipinski.NumHDonors(mol),
    "Drug_HBA": Lipinski.NumHAcceptors(mol),
    "Drug_RB": Lipinski.NumRotatableBonds(mol),
    "Drug_TPSA": rdMolDescriptors.CalcTPSA(mol),
    "Drug_Aromatic_Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    "Drug_Fraction_Csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    "Drug_Heavy_Atoms": mol.GetNumHeavyAtoms(),
}

# Путь сохранения — рядом с этим скриптом
output_path = Path(__file__).resolve().parent / "clotrimazole_descriptors.csv"

# Сохраняем результаты
with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=descriptors.keys())
    writer.writeheader()
    writer.writerow(descriptors)

print("Дескрипторы клотримазола:")
for name, value in descriptors.items():
    print(f"{name}: {value}")

print(f"\nРезультаты сохранены в: {output_path}")
