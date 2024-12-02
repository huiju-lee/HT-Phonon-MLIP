import pickle as pk
from pymatgen.io.ase import AseAtomsAdaptor

# Load data from pickle file.
pickle_file = './training_data.pickle'
with open(pickle_file, 'rb') as file:
    data = pk.load(file)

# Initialize lists to store extracted data.
structures = []
energies = []
forces = []
stresses = []

# Loop over dictionary keys and collect structures, energies, forces, and stresses.
for key in data.keys():
    structures.append(data[key][0])
    energies.append(data[key][1])
    forces.append(data[key][2])
    stresses.append(data[key][3])
        
for structure in structures:
    atoms = AseAtomsAdaptor().get_atoms(structure) # Convert pymatgen structure to ASE atoms

    # Append each structure to the output XYZ file in extended XYZ format.
    with open("./training_data.xyz", mode='a') as file:
        atoms.write(file, format='extxyz')

        
# Read back the XYZ file to add properties.
with open("./training_data.xyz", 'r') as file:
    lines = file.readlines()

    
i = 0 # Index for looping through lines in the XYZ file
count = 0 # Count of processed structures

while i < len(lines):
    # Check for the line defining atomic properties to add custom attributes.
    if "Properties=species:S:1:pos:R:3" in lines[i]:
        # Stop if processed all structures.
        if count >= len(structures):
            break

        # Add forces, energy, and stress attributes.
        lines[i] = lines[i].replace("Properties=species:S:1:pos:R:3", "Properties=species:S:1:pos:R:3:DFT_forces:R:3")
        new_line = lines[i].strip()

        # Create a formatted string for the stress tensor.
        stress_string = ' '.join([' '.join(map(str, stress)) for stress in stresses[count]])
        new_line += f" DFT_energy={energies[count]} DFT_stress=\"{stress_string}\"\n"
        lines[i] = new_line

        # Append forces to the lines for each atom in the structure.
        if count < len(forces):
            for j, line in enumerate(lines[i + 1:i + len(forces[count]) + 1]):
                line = line.strip()
                add_forces = f" {' '.join(['%16.8f' % force for force in forces[count][j]])}\n"
                lines[i + 1 + j] = line + add_forces

        count += 1 # Move to the next structure
    i += 1 # Move to the next line

# Write the modified lines back to the XYZ file.
with open("./training_data.xyz", "w") as file:
    file.writelines(lines)
