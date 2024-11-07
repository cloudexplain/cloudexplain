import sys
from typing import TypedDict

class ModelMetadata(TypedDict):
    model_name: str
    model_description: str
    model_version: str
    model_framework: str
    model_framework_version: str
    model_runtime: str
    model_runtime_version: str
    model_output_dimension: int
    model_input_dimension: int
    feature_names: list[str]

def create_model_metadata(model: object,
                          X,
                          y,
                          model_version: str=None,
                          model_description: str = None) -> ModelMetadata:
    model_class_name = model.__class__.__name__
    model_module = model.__class__.__module__

    if "sklearn" in model_module:
        import sklearn
        model_metadata = {
            "model_name": model_class_name,
            "model_description": model_description,
            "model_version": model_version,
            # "model_type": model_type,  # todo: check if regression, classification, multi-class, etc.
            "model_framework": "sklearn",
            "model_framework_version": sklearn.__version__,
            "model_runtime": "python",
            "model_runtime_version": sys.version,
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
            # todo: change this if this is a more difficult model
            "feature_names": list(X.columns),
        }
    elif "xgboost" in model_module:
        import xgboost
        model_metadata = {
            "model_name": model_class_name,
            "model_description": model_description,
            "model_version": model_version,
            # "model_type": model_type,  # todo: check if regression, classification, multi-class, etc.
            "model_framework": "xgboost",
            "model_framework_version": xgboost.__version__,
            "model_runtime": "python",
            "model_runtime_version": sys.version,
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
        }
    elif "catboost" in model_module:
        import catboost
        model_metadata = {
            "model_name": model_class_name,
            "model_description": model_description,
            "model_version": model_version,
            # "model_type": model_type,  # todo: check if regression, classification, multi-class, etc.
            "model_framework": "catboost",
            "model_framework_version": catboost.__version__,
            "model_runtime": "python",
            "model_runtime_version": sys.version,
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
        }
    elif "lightgbm" in model_module:
        import lightgbm
        model_metadata = {
            "model_name": model_class_name,
            "model_description": model_description,
            "model_version": model_version,
            # "model_type": model_type,  # todo: check if regression, classification, multi-class, etc.
            "model_framework": "lightgbm",
            "model_framework_version": lightgbm.__version__,
            "model_runtime": "python",
            "model_runtime_version": sys.version,
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
        }
    else:
        raise Exception(f"Model framework not supported: {model_module}")
    return model_metadata