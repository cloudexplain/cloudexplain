from pathlib import Path
from azure.storage.filedatalake import FileSystemClient

def get_or_create_folders_recursively(path: str | Path, file_system_client: FileSystemClient):
    folders = path.split("/")
    current_folder = file_system_client.get_directory_client(folders[0])
    if not current_folder.exists():
        current_folder.create_directory()
    for folder in folders[1:]:
        # current_folder.create_sub_directory(folder)
        current_folder = current_folder.get_sub_directory_client(folder)
        if not current_folder.exists():
            current_folder.create_directory()
    return current_folder