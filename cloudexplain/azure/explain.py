from azure.identity import AzureCliCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import FileSystemClient
from azure.storage.filedatalake import DataLakeServiceClient
from cloudexplain.create_model_metadata import create_model_metadata
import uuid
import pickle
import json
import asyncio
import threading
from typing import Union, Optional
import logging
import hashlib
import uuid
import requests
from cloudexplain.azure.utils import get_or_create_folders_recursively
from typing import Literal
from enum import Enum
import os

credentials = AzureCliCredential()

class MLType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    MULTI_OUTPUT_REGRESSION = "multi_output_regression"
    TEXT_GENERATION = "text_generation"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"

NAME_PATTERN_RG = "cloudexplain"
NAME_PATTERN_STORAGE_ACC = "cloudexplainmodels"
DATA_CONTAINER_NAME = "cloudexplaindata"
MODEL_CONTAINER_NAME = "cloudexplainmodels"

def get_subscription_id(credentials):
    """Get the subscription id of the current subscription.

    Args:
        credentials (_type_): _description_

    Returns:
        _type_: _description_
    """
    subscription_client = SubscriptionClient(credentials)

    # Get the list of subscriptions
    subscriptions = subscription_client.subscriptions.list()

    # Return the first enabled subscription
    for subscription in subscriptions:
        if subscription.state == 'Enabled':
            return subscription.subscription_id

def _find_cloudexplain_resource_group(credentials, resource_group_name: str = None):
    subscription_id = get_subscription_id(credentials=credentials)
    client = ResourceManagementClient(credentials, subscription_id=subscription_id)
    pattern_to_search = resource_group_name or NAME_PATTERN_RG

    for item in client.resource_groups.list():
        if pattern_to_search in item.name:
            return item

def _find_cloudexplain_storage_acc(subscription_id: str, credentials, cloudexplain_rg: str):
    storage_client = StorageManagementClient(credentials, subscription_id=subscription_id)
    # List storage accounts in the specified resource group
    storage_accounts = storage_client.storage_accounts.list_by_resource_group(cloudexplain_rg.name)

    # Print the storage account names
    for account in storage_accounts:
        if NAME_PATTERN_STORAGE_ACC in account.name:
            return account

def _get_data_container_client_from_account(credentials, account: str, container_name: str):
    blob_service_client = BlobServiceClient(account_url=f"https://{account.name}.blob.core.windows.net", credential=credentials)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client

def find_storage_account_name(resource_group_name: str | None):
    credentials = AzureCliCredential()

    cloudexplain_rg = _find_cloudexplain_resource_group(credentials=credentials, resource_group_name=resource_group_name)
    subscription_id = get_subscription_id(credentials=credentials)

    account = _find_cloudexplain_storage_acc(subscription_id=subscription_id, credentials=credentials, cloudexplain_rg=cloudexplain_rg)
    if account is None:
        raise Exception(f"No storage account with name {NAME_PATTERN_STORAGE_ACC} found. Please specify a resource_group using the argument {resource_group_name}.")
    return account

def get_container_client(container_name: str, resource_group_name: str | None = None) -> BlobServiceClient:
    """Get the container client for the cloudexplaindata container.

    Returns:
        BlobServiceClient: blob service client for the cloudexplaindata container.
    """
    account = find_storage_account_name(resource_group_name=resource_group_name)
    data_container_client = _get_data_container_client_from_account(credentials=credentials, account=account, container_name=container_name)
    return data_container_client


def get_file_system_client_from_account(resource_group_name: str | None) -> FileSystemClient:
    credentials = AzureCliCredential()

    # todo: create singletons for resource group, subscription id, and storage account
    cloudexplain_rg = _find_cloudexplain_resource_group(credentials=credentials, resource_group_name=resource_group_name)
    subscription_id = get_subscription_id(credentials=credentials)

    account = _find_cloudexplain_storage_acc(subscription_id=subscription_id, credentials=credentials, cloudexplain_rg=cloudexplain_rg)

    file_system_client = FileSystemClient(account_url=f"https://{account.name}.dfs.core.windows.net",
                                          credential=credentials,
                                          file_system_name=MODEL_CONTAINER_NAME
                                          )
    return file_system_client

async def _upload_create_folder_files_async(container_client, directory_name, data, file_name):
    """Upload data to a file in a directory, creating the directory if it does not exist. Currently unused."""
    directory_client = get_or_create_folders_recursively(directory_name, container_client)
    file_client = directory_client.get_file_client(file_name)
    file_client.upload_data(data, overwrite=True, encoding='utf-8')


async def _upload_blob_async(container_client, directory_name, data, file_name):
    container_client.upload_blob(f"{directory_name}/{file_name}", data, overwrite=True, encoding='utf-8')

async def _upload_files_async(data_container_client,
                              model_container_client,
                              directory_name,
                              X,
                              dumped_model,
                              model_metadata: Optional[dict],
                              run_metadata,
                              y=None,
                              baseline_data=None,
                              observation_id_column=None,
                              ):
    """Currently unused."""
    jobs = [_upload_create_folder_files_async(container_client=data_container_client, directory_name=directory_name, file_name="data.pickle", data=pickle.dumps((X, y))),
            ]
    if run_metadata["run_mode"] == "training":
        model_name = model_metadata["model_name"]
        model_version = model_metadata["model_version"]
        jobs.extend([_upload_create_folder_files_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model.pickle", data=dumped_model),
                     _upload_create_folder_files_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model_metadata.json", data=json.dumps(model_metadata, ensure_ascii=False))
                     ]
                    )
    if observation_id_column is not None:
        jobs.append(_upload_create_folder_files_async(container_client=data_container_client, directory_name=directory_name, file_name="observation_id_column.pickle", data=pickle.dumps(observation_id_column)))
    if baseline_data is not None:
        jobs.append(_upload_create_folder_files_async(container_client=data_container_client, directory_name=directory_name, file_name="baseline_data.pickle", data=pickle.dumps(baseline_data)))

    await asyncio.gather(
        *jobs
    )
    await _upload_create_folder_files_async(container_client=data_container_client, directory_name=directory_name, file_name="run_metadata.json", data=json.dumps(run_metadata, ensure_ascii=False))


async def _upload_blobs_async(data_container_client,
                              model_container_client,
                              directory_name,
                              X,
                              dumped_model,
                              model_metadata: Optional[dict],
                              run_metadata,
                              y=None,
                              baseline_data=None,
                              observation_id_column=None,
                              ):
    jobs = [_upload_blob_async(container_client=data_container_client, directory_name=directory_name, file_name="data.pickle", data=pickle.dumps((X, y))),
            ]
    if run_metadata["run_mode"] == "training":
        model_name = model_metadata["model_name"]
        model_version = model_metadata["model_version"]
        jobs.extend([_upload_blob_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model.pickle", data=dumped_model),
                     _upload_blob_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model_metadata.json", data=json.dumps(model_metadata, ensure_ascii=False))
                     ]
                    )
    if observation_id_column is not None:
        jobs.append(_upload_blob_async(container_client=data_container_client, directory_name=directory_name, file_name="observation_id_column.pickle", data=pickle.dumps(observation_id_column)))
    if baseline_data is not None:
        jobs.append(_upload_blob_async(container_client=data_container_client, directory_name=directory_name, file_name="baseline_data.pickle", data=pickle.dumps(baseline_data)))

    await asyncio.gather(
        *jobs
    )
    await _upload_blob_async(container_client=data_container_client, directory_name=directory_name, file_name="run_metadata.json", data=json.dumps(run_metadata, ensure_ascii=False))

def list_sub_directories(directory_client):
    sub_directories = [path.name for path in directory_client.get_paths() if path.is_directory]
    return sub_directories

def get_container_client_from_sas(sas_url: str, container_name: str) -> BlobServiceClient:
    """Get a container client from a SAS URL."""
    # Convert Data Lake SAS URL to Blob Storage SAS URL
    sas_url = sas_url.replace('.dfs.core.windows.net', '.blob.core.windows.net')

    blob_service_client = BlobServiceClient(account_url=sas_url, credential=None)
    return blob_service_client.get_container_client(container_name)

def get_file_system_client_from_sas(sas_url: str, container_name: str) -> FileSystemClient:
    """Get a file system client from a SAS URL for Data Lake operations."""
    # Ensure we're using the Data Lake endpoint
    if '.blob.core.windows.net' in sas_url:
        sas_url = sas_url.replace('.blob.core.windows.net', '.dfs.core.windows.net')
    
    datalake_service_client = DataLakeServiceClient(account_url=sas_url, credential=None)
    return datalake_service_client.get_file_system_client(container_name)

class ExplainModelContext:
    def __init__(self,
                 model,
                 X: Union["pandas.DataFrame", "numpy.ndarray"],
                 ml_type: Literal["binary_classification",
                                         "multiclass_classification",
                                         "regression",
                                         "multi_output_regression",
                                         "text_generation",
                                         "image_classification",
                                         "image_generation"] | MLType,
                 model_name: str,
                 model_version: str,
                 explanation_name: str,
                 data_source: str,
                 y: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
                 model_description: str = None,
                 resource_group_name: Optional[str] = None,
                 explanation_env: Optional[str] = "prod",
                 observation_id_column: Optional[Union[list[Union[int, str]], "pandas.Series", "numpy.ndarray"]] = None,
                 observation_entity: Optional[str] = None,
                 output_unit: Optional[str] = None,
                 feature_units: Optional[dict[str, str]] = None,
                 is_higher_output_better: Union[bool, tuple[bool]] = True,
                 feature_descriptions: Optional[dict[str, str]] = None,
                 baseline_data: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
                 function_url: Optional[str] = None,
                 api_token: Optional[str] = None,
                 ):
        try:
            if isinstance(ml_type, str):
                ml_type = MLType(ml_type)
        except ValueError as e:
            raise ValueError(f"ml_type must be one of {', '.join([ml_type.value for ml_type in MLType])}.") from e
        if resource_group_name is None:
            self.api_token = api_token or os.environ["CLOUDEXPLAIN_API_TOKEN"]
            self.function_url = function_url or os.environ["CLOUDEXPLAIN_FUNCTION_URL"]
            self.api_upload = True
        else:
            self.api_upload = False
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.model_version = model_version
        self.model_description = model_description
        self.ml_type = ml_type.value

        self.observation_entity = observation_entity
        self.output_unit =  output_unit
        self.feature_units = feature_units
        self.is_higher_output_better = is_higher_output_better
        self.feature_descriptions = feature_descriptions
        if feature_descriptions is None:
            self.feature_display_name_map= dict()
            self.feature_description_map = dict()
        else:
            self.feature_display_name_map = {k: v['display_description'] for k, v in feature_descriptions.items() if v.get("display_description")}
            self.feature_description_map = {k: v['description'] for k, v in feature_descriptions.items() if v.get("description")}
        self.feature_descriptions = feature_descriptions
        self.baseline_data = baseline_data

        # get container clients -> todo: use file clients instead
        if self.api_upload:
            response = requests.post(self.function_url, headers={"Content-Type": "application/json"}, json={"token": self.api_token})
            response.raise_for_status()
            
            # Parse the JSON response correctly
            response_data = response.json()
            data_sas_url = response_data["data_sas_url"]

            self.data_container_client = get_container_client_from_sas(sas_url=data_sas_url, container_name=DATA_CONTAINER_NAME)
            self.model_container_client = get_container_client_from_sas(sas_url=data_sas_url, container_name=MODEL_CONTAINER_NAME)
            
            # Create file system client from SAS URL for Data Lake operations
            self.fsc = get_file_system_client_from_sas(sas_url=data_sas_url, container_name=MODEL_CONTAINER_NAME)
        else:
            self.data_container_client = get_container_client(resource_group_name=resource_group_name, container_name=DATA_CONTAINER_NAME)
            self.model_container_client = get_container_client(resource_group_name=resource_group_name, container_name=MODEL_CONTAINER_NAME)
            self.fsc = get_file_system_client_from_account(resource_group_name=resource_group_name)
            
        # Check model folder existence - this code remains the same for both paths
        model_folder_client = get_or_create_folders_recursively(f"{model_name}/{model_version}", self.fsc)
        if model_folder_client.get_file_client("model_metadata.json").exists() and y is not None:
            existing_versions = [int(version.name) for version in list_sub_directories(model_folder_client)]
            raise Exception(f"Model with name {model_name} and version {model_version} already exists and you are running in training mode. "
                            "To run in inference mode, please don't hand over `y`, alternatively choose a different model version. "
                            )
                            # todo: this does not work, we need to go one level higher and list the folders there
                            # f"These versions exist already: {existing_versions}")
        self.run_uuid = str(uuid.uuid4())
        logging.info(f"Starting explanation run with uuid {self.run_uuid}")
        self.directory_name = f"explanation_{self.run_uuid}"
        self.model_metadata = None 
        self.data_source = data_source
        self.explanation_env = explanation_env
        self.explanation_name = explanation_name
        self.observation_id_column = observation_id_column
        self.run_metadata = None
        self.upload_thread = None

    def __enter__(self):
        # Start the upload in a separate thread
        self.upload_thread = threading.Thread(target=self._start_upload)
        self.upload_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Wait for the upload to complete
        if self.upload_thread:
            self.upload_thread.join()

    def _start_upload(self):
        # create model hash
        # todo: only hash pickle dump model once
        dumped_model = pickle.dumps(self.model)
        model_hash = str(uuid.UUID(hashlib.md5(dumped_model).hexdigest()))
        self.run_metadata = {"data_source": self.data_source,
                             "observation_id_column_provided": True if self.observation_id_column is not None else False,
                             "explanation_env": self.explanation_env,
                             "explanation_name": self.explanation_name,
                             "X_shape": self.X.shape,
                             "run_uuid": self.run_uuid,
                             "run_mode": "inference" if self.y is None else "training",
                             "model_name": self.model_name,
                             "model_version": self.model_version,
                             "observation_entity": self.observation_entity,
                             "output_unit": self.output_unit,
                             "feature_units": self.feature_units,
                             "is_higher_output_better": self.is_higher_output_better,
                             "feature_display_name_map": self.feature_display_name_map,
                             "feature_description_map": self.feature_description_map,
                             "feature_descriptions": self.feature_descriptions
                             }

        if self.run_metadata["run_mode"] == "training":
            self.model_metadata = create_model_metadata(self.model,
                                                        self.X,
                                                        self.y,
                                                        model_name=self.model_name,
                                                        model_version=self.model_version,
                                                        model_description=self.model_description,
                                                        model_hash=model_hash,
                                                        ml_type=self.ml_type
                                                        )
        asyncio.run(_upload_blobs_async(
                                        data_container_client=self.data_container_client,
                                        model_container_client=self.model_container_client,
                                        directory_name=self.directory_name,
                                        X=self.X,
                                        y=self.y,
                                        baseline_data=self.baseline_data,
                                        dumped_model=dumped_model,
                                        model_metadata=self.model_metadata,
                                        run_metadata=self.run_metadata,
                                        observation_id_column=self.observation_id_column
                                        ))


def explain(model,
            X: Union["pandas.DataFrame", "numpy.ndarray"],
            model_name: str,
            model_version: str,
            explanation_name: str,
            data_source: str,
            ml_type: Literal["binary_classification",
                             "multiclass_classification",
                             "regression",
                             "multi_output_regression",
                             "text_generation",
                             "image_classification",
                             "image_generation"],
            y: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
            model_description: str = None,
            resource_group_name: str | None = None,
            explanation_env: str | None = "prod",
            observation_id_column: str | None = None,
            observation_entity: str | None = None,
            output_unit: str | None = None,
            feature_units: dict[str, str] | None = None,
            is_higher_output_better: bool | tuple[bool] = True,
            feature_descriptions: dict[str, str] | None = None,
            baseline_data: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
            function_url: Optional[str] = None,
            api_token: Optional[str] = None) -> ExplainModelContext:
    """Upload the model, data, and metadata to the cloudexplaindata container asynchronously.

    Usage:
    ```python
    import cloudexplain

    with cloudexplain.explain(model, X, y, model_version="1.0.0", model_description="This is a model") as model:
        result = model.fit(X, y)
        save_result(result)
    ```

    Args:
        model (Any): Any model that can be pickled and explained.
        X (_type_): _description_
        y (_type_): _description_
        model_name (str, optional): Name of the used model. Defaults to None.
        model_version (str, optional): _description_. Defaults to None.
        model_description (str, optional): _description_. Defaults to None.
        resource_group_name (str, optional): _description_. Defaults to None.
        explanation_env (str, optional): The environment in which the explanation takes place. Typicall for model development one chooses "dev", for productive runs "prod". Defaults to "prod".
        explanation_name (str, optional): The name of the explanation. Under this name the explanation will be stored in the database and be viewable in the cloudexplain dashboard. Defaults to None.
        data_source (str, optional): The source of the data. Runs on the same source can be compared against each other. Defaults to None.
        observation_id_column (str, optional): A column in X that refers that marks a unique identifier of the observation/row. If provided the explanation of the given row will be accessible by this id. Defaults to None.
        ml_type (Literal, optional): The type of the machine learning model. Defaults to None. One of "binary_classification", "multiclass_classification", "regression", "multi_output_regression", "text_generation", "image_classification", "image_generation".
        observation_entity: str, optional
            The entity of an observation. Answers the question what an observation represents. For instance for a churn use case this might be "customer", for the california housing dataset "house", for iris "plant". This can be None if not applicable or required.
        output_unit: str, optional
            The unit of measurement for the output (e.g., percent, dollars, meters, pounds). If the output is a simple count, we advise to leave this empty. This can be None if there is no specific unit.
        feature_units: dict[str, str], optional
            A dictionary mapping feature names to their respective units of measurement. For example, {'feature1': 'meters', 'feature2': 'dollars', 'feature3': 'months'}. This can be None.
        is_higher_output_better : bool, tuple[bool]
            A boolean (or tuple of boolean in case of multi-output models) indicating whether higher values of the output are considered better (defaults to True). In case of multi output models this should be a tuple of booleans, e.g. (true, false, true).
        feature_descriptions : dict[str, str], optional
            A dictionary providing descriptions for each feature. This can help in understanding the role and details of each feature.
            For example, {'feature1': {'is_categorical': True, 'description': 'This feature represents ...', 'display_description': 'Describes correlation', "encoding": {"feature_val1": 0, "feature_val2": 1, ...}},
                          'feature2': {'is_categorical': False, ...}
                          }
            'is_categorical', 'description' and 'display_description' are mandatory fields.
        baseline_data: pandas.DataFrame, numpy.ndarray, optional
            The baseline data that is used to explain the model. This can be None.
        Returns:
            ExplainModelContext: The context manager that uploads the model, data, and metadata to the cloudexplaindata container asynchronously.
    """
    return ExplainModelContext(model=model,
                               X=X,
                               y=y,
                               model_name=model_name,
                               model_version=model_version,
                               model_description=model_description,
                               resource_group_name=resource_group_name,
                               explanation_env=explanation_env,
                               explanation_name=explanation_name,
                               data_source=data_source,
                               observation_id_column=observation_id_column,
                               ml_type=ml_type,
                               observation_entity=observation_entity,
                               output_unit=output_unit,
                               feature_units=feature_units,
                               is_higher_output_better=is_higher_output_better,
                               feature_descriptions=feature_descriptions,
                               baseline_data=baseline_data,
                               function_url=function_url,
                               api_token=api_token,
    )
