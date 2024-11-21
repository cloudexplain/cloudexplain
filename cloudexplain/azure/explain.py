from azure.identity import AzureCliCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import FileSystemClient
from cloudexplain.create_model_metadata import create_model_metadata
import uuid
import pickle
import json
import asyncio
import threading
from typing import Any, Union, Optional, Tuple
import logging
import hashlib
import uuid
from cloudexplain.azure.utils import get_or_create_folders_recursively

credentials = AzureCliCredential()


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
    directory_client = get_or_create_folders_recursively(directory_name, container_client)
    file_client = directory_client.get_file_client(file_name)
    file_client.upload_data(data, overwrite=True, encoding='utf-8')

async def _upload_files_async(container_client, directory_name, data, file_name):
    container_client.upload_blob(f"{directory_name}/{file_name}", data, overwrite=True, encoding='utf-8')

async def _upload_blobs_async(data_container_client,
                              model_container_client,
                              directory_name,
                              X,
                              dumped_model,
                              model_metadata: Optional[dict],
                              run_metadata,
                              y=None,
                              observation_id_column=None,
                              ):
    jobs = [_upload_files_async(container_client=data_container_client, directory_name=directory_name, file_name="data.pickle", data=pickle.dumps((X, y))),
            ]
    if run_metadata["run_mode"] == "training":
        model_name = model_metadata["model_name"]
        model_version = model_metadata["model_version"]
        jobs.extend([_upload_create_folder_files_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model.pickle", data=dumped_model),
                     _upload_create_folder_files_async(container_client=model_container_client, directory_name=f"{model_name}/{model_version}", file_name="model_metadata.json", data=json.dumps(model_metadata))
                     ]
                    )
    if observation_id_column is not None:
        jobs.append(_upload_files_async(container_client=data_container_client, directory_name=directory_name, file_name="observation_id_column.pickle", data=pickle.dumps(observation_id_column)))

    await asyncio.gather(
        *jobs
    )
    await _upload_files_async(container_client=data_container_client, directory_name=directory_name, file_name="run_metadata.json", data=json.dumps(run_metadata))

def list_sub_directories(directory_client):
    sub_directories = [path.name for path in directory_client.get_paths() if path.is_directory]
    return sub_directories


class ExplainModelContext:
    def __init__(self,
                 model,
                 X: Union["pandas.DataFrame", "numpy.ndarray"],
                 y: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
                 model_name: str | None = None,
                 model_version=None,
                 model_description=None,
                 resource_group_name: str | None = None,
                 explanation_env: str | None = "prod",
                 explanation_name: str | None = None,
                 data_source: str | None = None,
                 observation_id_column: Optional[Union[list[Union[int, str]], "pandas.Series", "numpy.ndarray"]] = None):
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.model_version = model_version
        self.model_description = model_description

        # get container clients -> todo: use file clients instead
        self.data_container_client = get_container_client(resource_group_name=resource_group_name, container_name=DATA_CONTAINER_NAME)
        self.model_container_client = get_container_client(resource_group_name=resource_group_name, container_name=MODEL_CONTAINER_NAME)

        self.fsc = get_file_system_client_from_account(resource_group_name=resource_group_name)
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
                             "run_uuid": self.run_uuid,
                             "run_mode": "inference" if self.y is None else "training",
                             "model_name": self.model_name,
                             "model_version": self.model_version,
                             }

        if self.run_metadata["run_mode"] == "training":
            self.model_metadata = create_model_metadata(self.model,
                                                        self.X,
                                                        self.y,
                                                        model_name=self.model_name,
                                                        model_version=self.model_version,
                                                        model_description=self.model_description,
                                                        model_hash=model_hash,
                                                        )
        asyncio.run(_upload_blobs_async(data_container_client=self.data_container_client,
                                        model_container_client=self.fsc,
                                        directory_name=self.directory_name,
                                        X=self.X,
                                        y=self.y,
                                        dumped_model=dumped_model,
                                        model_metadata=self.model_metadata,
                                        run_metadata=self.run_metadata,
                                        observation_id_column=self.observation_id_column
                                        )
                    )

def explain(model,
            X: Union["pandas.DataFrame", "numpy.ndarray"],
            y: Optional[Union["pandas.DataFrame", "numpy.ndarray"]] = None,
            model_name: str = None,
            model_version: str = None,
            model_description: str = None,
            resource_group_name: str | None = None,
            explanation_env: str | None = "prod",
            explanation_name: str | None = None,
            data_source: str | None = None,
            observation_id_column: str | None = None) -> ExplainModelContext:
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

    Returns:
        _type_: _description_
    """
    return ExplainModelContext(model,
                               X,
                               y,
                               model_name=model_name,
                               model_version=model_version,
                               model_description=model_description,
                               resource_group_name=resource_group_name,
                               explanation_env=explanation_env,
                               explanation_name=explanation_name,
                               data_source=data_source,
                               observation_id_column=observation_id_column)

