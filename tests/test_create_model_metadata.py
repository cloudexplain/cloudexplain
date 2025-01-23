from cloudexplain.create_model_metadata import create_model_metadata
import shap

def test_create_model_metadata_xgboost():
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model_name = "XGBoost model"
    model_description = "A simple xgboost model"
    model_version = "0.1.0"
    ml_type = "classification"

    X, y = shap.datasets.adult(n_points=100)
    result = create_model_metadata(model=model,
                          X=X,
                          y=y,
                          model_version=model_version,
                          model_description=model_description,
                          model_name=model_name,
                          ml_type=ml_type,
                          )

    expected = {
            "model_name": model_name,
            "model_type": "XGBClassifier",
            "model_description": model_description,
            "model_version": model_version,
            "model_runtime": "python",
            # "model_runtime_version": sys.version,  # ignore
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
            # todo: change this if this is a more difficult model
            "feature_names": list(X.columns),
            # "model_hash": model_hash,  # ignore
            "ml_type": ml_type,
            "model_framework": "xgboost",
    }

    result.pop("model_runtime_version")
    result.pop("model_hash")
    result.pop("model_framework_version")
    assert result == expected


def test_create_model_metadata_catboost():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model_name = "Catboost model"
    model_description = "A simple catboost model"
    model_version = "0.1.0"
    ml_type = "classification"

    X, y = shap.datasets.adult(n_points=100)
    result = create_model_metadata(model=model,
                                   X=X,
                                   y=y,
                                   model_version=model_version,
                                   model_description=model_description,
                                   model_name=model_name,
                                   ml_type=ml_type,
                                   )

    expected = {
            "model_name": model_name,
            "model_type": "CatBoostClassifier",
            "model_description": model_description,
            "model_version": model_version,
            "model_runtime": "python",
            # "model_runtime_version": sys.version,  # ignore
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
            # todo: change this if this is a more difficult model
            "feature_names": list(X.columns),
            # "model_hash": model_hash,  # ignore
            "ml_type": ml_type,
            "model_framework": "catboost",
    }

    result.pop("model_runtime_version")
    result.pop("model_hash")
    result.pop("model_framework_version")
    assert result == expected


def test_create_model_metadata_lgbm_classifier():
    from lightgbm import LGBMClassifier

    model = LGBMClassifier()
    model_name = "LightGBM model"
    model_description = "A simple lightgbm model"
    model_version = "0.1.0"
    ml_type = "classification"

    X, y = shap.datasets.adult(n_points=100)
    model.fit(X, y)
    result = create_model_metadata(model=model,
                                   X=X,
                                   y=y,
                                   model_version=model_version,
                                   model_description=model_description,
                                   model_name=model_name,
                                   ml_type=ml_type,
                                   )

    expected = {
            "model_name": model_name,
            "model_type": "LGBMClassifier",
            "model_description": model_description,
            "model_version": model_version,
            "model_runtime": "python",
            # "model_runtime_version": sys.version,  # ignore
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
            # todo: change this if this is a more difficult model
            "feature_names": list(X.columns),
            # "model_hash": model_hash,  # ignore
            "ml_type": ml_type,
            "model_framework": "lightgbm",
    }

    result.pop("model_runtime_version")
    result.pop("model_hash")
    result.pop("model_framework_version")
    assert result == expected


def test_create_model_metadata_lgbm_regressor():
    from lightgbm import LGBMRegressor

    model = LGBMRegressor()
    model_name = "LightGBM model"
    model_description = "A simple lightgbm model"
    model_version = "0.1.0"
    ml_type = "regression"

    X, y = shap.datasets.adult(n_points=100)
    result = create_model_metadata(model=model,
                                   X=X,
                                   y=y,
                                   model_version=model_version,
                                   model_description=model_description,
                                   model_name=model_name,
                                   ml_type=ml_type,
                                   )

    expected = {
            "model_name": model_name,
            "model_type": "LGBMRegressor",
            "model_description": model_description,
            "model_version": model_version,
            "model_runtime": "python",
            # "model_runtime_version": sys.version,  # ignore
            "model_output_dimension": y.shape[1] if len(y.shape) > 1 else 1,
            "model_input_dimension": X.shape[-1],
            # todo: change this if this is a more difficult model
            "feature_names": list(X.columns),
            # "model_hash": model_hash,  # ignore
            "ml_type": ml_type,
            "model_framework": "lightgbm",
    }

    result.pop("model_runtime_version")
    result.pop("model_hash")
    result.pop("model_framework_version")
    assert result == expected