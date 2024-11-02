from azure.identity import AzureCliCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient

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

def _find_cloudexplain_resource_group(credentials):
    subscription_id = get_subscription_id(credentials=credentials)
    client = ResourceManagementClient(credentials, subscription_id=subscription_id)

    for item in client.resource_groups.list():
        if NAME_PATTERN_RG in item.name:
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

def get_data_container_client() -> BlobServiceClient:
    """Get the container client for the cloudexplaindata container.

    Returns:
        BlobServiceClient: blob service client for the cloudexplaindata container.
    """
    credentials = AzureCliCredential()

    cloudexplain_rg = _find_cloudexplain_resource_group(credentials=credentials)
    subscription_id = get_subscription_id(credentials=credentials)

    account = _find_cloudexplain_storage_acc(subscription_id=subscription_id, credentials=credentials, cloudexplain_rg=cloudexplain_rg)
    container_client = _get_data_container_client_from_account(credentials=credentials, account=account)
    return container_client