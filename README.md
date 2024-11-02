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


#### Troubleshooting
When logged in with multiple accounts it can be the case that the permissions are not handled correctly. Signing out of all users
except for the one who has the correct rights helps.
```bash
az logout
```