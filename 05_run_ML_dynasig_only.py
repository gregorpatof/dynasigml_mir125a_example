from dynasigml.dynasig_ml_model import DynaSigML_Model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import json


def hard_test_dynasig():
    with open("test_ids_hard.json") as f:
        test_ids = json.load(f)
    with open("train_ids_hard.json") as f:
        train_ids = json.load(f)
    dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids, train_ids=train_ids, verbose=True,
                                 predictor_columns=['dynasig'],
                                 ml_models=[RandomForestRegressor(),
                                            GradientBoostingRegressor()],
                                 ml_models_labels=["RF", "GBR"], measured_property="maturation efficiency")
    dsml_model.train_test_lasso()
    dsml_model.train_test_ml_models()
    dsml_model.performance_report()
    dsml_model.save_to_file('dsml_model_hard_dynasig')
    dsml_model.make_graphs('graphs_hard_dynasig')


def inverted_test_dynasig():
    with open("test_ids_inverted.json") as f:
        test_ids = json.load(f)
    with open("train_ids_inverted.json") as f:
        train_ids = json.load(f)
    dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids, train_ids=train_ids, verbose=True,
                                 predictor_columns=['dynasig'],
                                 ml_models=[RandomForestRegressor(),
                                            GradientBoostingRegressor()],
                                 ml_models_labels=["RF", "GBR"], measured_property="maturation efficiency")
    dsml_model.train_test_lasso()
    dsml_model.train_test_ml_models()
    dsml_model.performance_report()
    dsml_model.save_to_file('dsml_model_inverted_dynasig')
    dsml_model.make_graphs('graphs_inverted_dynasig')


if __name__ == "__main__":
    hard_test_dynasig()
    inverted_test_dynasig()
