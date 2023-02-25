from dynasigml.dynasig_ml_model import load_dynasigml_model_from_file

if __name__ == "__main__":
    dsml_model = load_dynasigml_model_from_file("dsml_model_inverted.pickle")
    dsml_model.map_coefficients("mir125a_variants/mir125a_WT.pdb", "coefficients_inverted.pdb")

    dsml_model_hard = load_dynasigml_model_from_file("dsml_model_hard.pickle")
    dsml_model_hard.map_coefficients("mir125a_variants/mir125a_WT.pdb", "coefficients_hard.pdb")
