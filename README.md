### Cloudexplain

This is the open source cloudexplain package (or what it is going to be). This just uploads your current data to the azure account you are currently logged in via az.

#### Azure
To install this for azure run:
```bash
pip install cloudexplain[azure]
```
In order to run it on azure you must have write rights on the `cloudexplainmodels` storage account on the resource group `cloudexplain` on your subscription and must be logged in via
```bash
az login
```

#### Example Usage

```python
import shap
import sklearn
import cloudexplain

X, y = shap.datasets.adult(n_points=100)
X["adult_id"] = list(range(len(X)))
dtc = sklearn.tree.DecisionTreeClassifier()
dtc.fit(X, y)


with cloudexplain.azure.explain(dtc,
                                X,
                                y,
                                model_version="0.0.1",
                                model_description="This is a new model.",
                                resource_group_name="mycomp-cloudexplain-tf",
                                explanation_name="dummy_name",
                                explanation_env="dev",
                                data_source="shap_adult_dataset",
                                observation_id_column="adult_id") as run:
    print("This is the run uuid", run.run_uuid)
    pred = dtc.predict(X)
```


#### Troubleshooting
When logged in with multiple accounts it can be the case that the permissions are not handled correctly. Signing out of all users
except for the one who has the correct rights helps.
```bash
az logout
```