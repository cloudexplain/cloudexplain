from azure.identity import AzureCliCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from cloudexplain.create_model_metadata import create_model_metadata
import uuid
import pickle
import json
import asyncio
import threading
from typing import Any
import logging

credentials = AzureCliCredential()


NAME_PATTERN_RG = "cloudexplain"
NAME_PATTERN_STORAGE_ACC = "cloudexplainmodels"
CONTAINER_NAME = "cloudexplaindata"

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

def _get_data_container_client_from_account(credentials, account: str):
    blob_service_client = BlobServiceClient(account_url=f"https://{account.name}.blob.core.windows.net", credential=credentials)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    return container_client

def get_data_container_client(resource_group_name: str | None) -> BlobServiceClient:
    """Get the container client for the cloudexplaindata container.

    Returns:
        BlobServiceClient: blob service client for the cloudexplaindata container.
    """
    credentials = AzureCliCredential()

    cloudexplain_rg = _find_cloudexplain_resource_group(credentials=credentials, resource_group_name=resource_group_name)
    subscription_id = get_subscription_id(credentials=credentials)

    account = _find_cloudexplain_storage_acc(subscription_id=subscription_id, credentials=credentials, cloudexplain_rg=cloudexplain_rg)
    container_client = _get_data_container_client_from_account(credentials=credentials, account=account)
    return container_client

async def _upload_files_async(container_client, directory_name, data, file_name):
    container_client.upload_blob(f"{directory_name}/{file_name}", data, overwrite=True, encoding='utf-8')

async def _upload_blobs_async(container_client, directory_name, X, y, model, model_metadata, run_metadata):
    await asyncio.gather(
        _upload_files_async(container_client=container_client, directory_name=directory_name, file_name="data.pickle", data=pickle.dumps((X, y))),
        _upload_files_async(container_client=container_client, directory_name=directory_name, file_name="model.pickle", data=pickle.dumps(model)),
        _upload_files_async(container_client=container_client, directory_name=directory_name, file_name="model_metadata.json", data=json.dumps(model_metadata))
    )
    await _upload_files_async(container_client=container_client, directory_name=directory_name, file_name="run_metadata.json", data=json.dumps(run_metadata))


class ExplainModelContext:
    def __init__(self,
                 model,
                 X,
                 y,
                 model_name: str | None = None,
                 model_version=None,
                 model_description=None,
                 resource_group_name: str | None = None,
                 explanation_env: str | None = "prod",
                 explanation_name: str | None = None,
                 data_source: str | None = None,
                 observation_id_column: str | None = None):
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.model_version = model_version
        self.model_description = model_description
        self.container_client = get_data_container_client(resource_group_name=resource_group_name)
        self.run_uuid = str(uuid.uuid4())
        logging.info(f"Starting explanation run with uuid {self.run_uuid}")
        self.directory_name = f"explanation_{self.run_uuid}"
        self.model_metadata = create_model_metadata(model,
                                                    X,
                                                    y,
                                                    model_name=model_name,
                                                    model_version=model_version,
                                                    model_description=model_description,
                                                    )
        self.run_metadata = {"data_source": data_source,
                             "observation_id_column": observation_id_column,
                             "explanation_env": explanation_env,
                             "explanation_name": explanation_name,
                             "run_uuid": self.run_uuid,
                             }
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
        asyncio.run(_upload_blobs_async(self.container_client, self.directory_name, self.X, self.y, self.model, self.model_metadata, self.run_metadata))

def explain(model,
            X,
            y,
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

