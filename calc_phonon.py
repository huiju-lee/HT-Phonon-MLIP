import os
import yaml
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import MACECalculator, mace_mp

class PhononCalculation:
    """
    Class to handle phonon calculations using machine learning interatomic potential (MLIP).
    """
    def __init__(self, supercell_matrix, structure_file, phonopy_params_yaml_file, FC_file, band_yaml_file):
        """
        Initialize the PhononCalculation with necessary files and parameters.
        """
        self.supercell_matrix = supercell_matrix
        self.structure_file = structure_file
        self.phonopy_params_yaml_file = phonopy_params_yaml_file
        self.FC_file = FC_file
        self.band_yaml_file = band_yaml_file

        try:
            # Load the crystal structure and initialize Phonopy object
            self.unitcell, _ = read_crystal_structure(self.structure_file, interface_mode='vasp')
            self.phonon = Phonopy(self.unitcell, supercell_matrix=self.supercell_matrix)
            self.phonon.generate_displacements(distance=0.03)
            self.supercells = self.phonon.supercells_with_displacements
            self.displacement_data = []
            self.displacement_dataset = self.phonon.get_displacement_dataset()
        except Exception as e:
            raise FileNotFoundError(f"Error loading structure file {self.structure_file}: {e}")

    def write_phonopy_params_yaml(self, calculator):
        """
        Generate the phonopy parameters YAML file by calculating forces
        for displaced supercells and saving the results.
        """
        for supercell in self.supercells:
            # Convert the structure into pymatgen format
            structure = Structure(
                lattice=supercell.cell,
                species=supercell.symbols[:],
                coords=supercell.scaled_positions
            )
            ase_atoms = AseAtomsAdaptor.get_atoms(structure)

            try:
                # Calculate forces using the trained potential model
                calculator.calculate(atoms=ase_atoms, properties='forces')
                forces = calculator.results["forces"]
                self.displacement_data.append({'structure': structure, 'forces': forces})
            except Exception as e:
                raise RuntimeError(f"Force calculation error for {structure}: {e}")

        # Add forces to the displacement dataset
        for i in range(len(self.supercells)):
            self.displacement_dataset['first_atoms'][i]['forces'] = self.displacement_data[i]['forces']

        # Save dataset to YAML file
        try:
            self.phonon.dataset = self.displacement_dataset
            self.phonon.produce_force_constants()
            self.phonon.save(filename=self.phonopy_params_yaml_file, settings={'force_constants': False})
        except Exception as e:
            raise IOError(f"Error saving phonopy params YAML file {self.phonopy_params_yaml_file}: {e}")

    def write_FORCE_CONSTANTS(self):
        """
        Write the force constants to the FORCE_CONSTANTS file.
        """
        try:
            self.phonon = phonopy.load(self.phonopy_params_yaml_file)
            phonopy_ifc = self.phonon.force_constants
            (natom1, natom2, dim1, dim2) = phonopy_ifc.shape
            ifc_str = str(natom1) + "\n"
            for i in range(natom1):
                for j in range(natom2):
                    ifc_str += str(i + 1) + " " + str(j + 1) + "\n"
                    mat_ifc = phonopy_ifc[i][j]
                    for line in mat_ifc:
                        formatted_line = "  ".join(f"{val:.15f}" for val in line)
                        ifc_str += f"{formatted_line}\n"
            with open(self.FC_file, "w") as file:
                file.write(ifc_str)
        except Exception as e:
            raise IOError(f"Error writing FORCE_CONSTANTS to {self.FC_file}: {e}")

    def write_band_yaml(self):
        """
        Write the band structure data to the YAML file.
        """
        try:
            self.phonon = phonopy.load(self.phonopy_params_yaml_file)
            self.phonon.auto_band_structure(write_yaml=True, filename=self.band_yaml_file)
        except Exception as e:
            raise IOError(f"Error writing band YAML file {self.band_yaml_file}: {e}")

    def write_thermal_properties_yaml(self, thermal_properties_yaml_file, mesh):
        """
        Write thermal properties to the YAML file.
        """
        try:
            self.phonon = phonopy.load(self.phonopy_params_yaml_file)
            self.phonon.run_mesh(mesh)
            self.phonon.run_thermal_properties(t_step=100, t_max=2000, t_min=100, cutoff_frequency=0.1)
            self.phonon.write_yaml_thermal_properties(filename=thermal_properties_yaml_file)
        except Exception as e:
            raise IOError(f"Error writing thermal properties YAML file {thermal_properties_yaml_file}: {e}")

def main():
    # Model setup for calculations
    model_path = '../model/MACE-F_swa.model'
    calculator = MACECalculator(model_path, device='cpu', default_dtype='float64')

    # Directory containing parameters YAML files
    params_yaml_dir = "../phonon_data/dft_params_yaml/"
    DFT_params_yaml = [
        file for file in os.listdir(params_yaml_dir) if file.endswith("phonopy_params.yaml")
    ]

    os.makedirs("ml_params_yaml", exist_ok=True)
    os.makedirs("ifc", exist_ok=True)
    os.makedirs("band_yaml", exist_ok=True)
    os.makedirs("thermal_properties_yaml", exist_ok=True)

    for params_yaml_file in DFT_params_yaml:
        formula, ICSD_number = params_yaml_file.split('-')[:2]
        print(f"{formula}-{ICSD_number}")

        # Load the supercell matrix from the YAML file
        with open(f"{params_yaml_dir}{params_yaml_file}", 'r') as file:
            yaml_data = yaml.safe_load(file)
            supercell_matrix = yaml_data.get('supercell_matrix')
        
        if supercell_matrix is None:
            raise ValueError(f"Missing supercell matrix in {params_yaml_file}")

        structure_file = f"../phonon_data/poscars/POSCAR-{formula}-{ICSD_number}"
        phonopy_params_yaml_file = f"./ml_params_yaml/phonopy_params.yaml-{formula}-{ICSD_number}"
        FC_file = f"./ifc/FORCE_CONSTANTS-{formula}-{ICSD_number}"
        band_yaml_file = f"./band_yaml/{formula}-{ICSD_number}-mace_band.yaml"
        thermal_properties_yaml_file = f"./thermal_properties_yaml/{formula}-{ICSD_number}-thermal_properties.yaml"

        # Initialize PhononCalculation instance and run methods
        phonon_calc = PhononCalculation(
            supercell_matrix, structure_file, phonopy_params_yaml_file, FC_file, band_yaml_file
        )
        phonon_calc.write_phonopy_params_yaml(calculator)
        phonon_calc.write_FORCE_CONSTANTS()
        phonon_calc.write_band_yaml()
        phonon_calc.write_thermal_properties_yaml(thermal_properties_yaml_file, mesh=75)

if __name__ == "__main__":
    main()
