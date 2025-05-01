import os
import subprocess

from typing import Any, Dict


class DBHandler:
    def __init__(self, config: Dict[str, Any]):
        self.root_path = config['root_path']
        self.host = config['remote_host']
        self.user = config['remote_user']
        self.data_path = config['remote_data_path']

    def list_dbs(self):
        path = os.path.join(self.root_path, "data/live_bot_databases")
        return [db for db in os.listdir(path) if db.endswith(".sqlite")]

    def fetch_dbs(self):
        local_path = os.path.join(self.root_path, "data/live_bot_databases")

        # Ensure the local directory exists
        os.makedirs(local_path, exist_ok=True)
        # Step 1: List directories in BACKEND_API_DATA_PATH
        list_dirs_cmd = f'ssh {self.user}@{self.host} "ls -d {self.data_path}/*/"'
        try:
            result = subprocess.run(list_dirs_cmd, shell=True, capture_output=True, text=True, check=True)
            directories = result.stdout.strip().split("\n")
        except subprocess.CalledProcessError as e:
            print(f"Error fetching directories: {e.stderr}")
            exit(1)

        # Step 2: Iterate through directories and find SQLite files
        for directory in directories:
            sequence_dir = directory.strip()
            data_folder = f"{sequence_dir}/data"

            # Find SQLite files, excluding "v2_with_controllers.sqlite"
            find_files_cmd = f'ssh {self.user}@{self.host} "find {data_folder} -type f -name \'*.sqlite\' ! -name \'v2_with_controllers.sqlite\'"'

            try:
                file_result = subprocess.run(find_files_cmd, shell=True, capture_output=True, text=True, check=True)
                sqlite_files = file_result.stdout.strip().split("\n")
            except subprocess.CalledProcessError as e:
                print(f"Error fetching SQLite files in {data_folder}: {e.stderr}")
                continue  # Skip this folder if there's an error

            # Step 3: Transfer SQLite files using SCP
            for remote_file in sqlite_files:
                if remote_file:  # Ignore empty results
                    scp_cmd = f"scp {self.user}@{self.host}:{remote_file} {local_path}/"
                    try:
                        subprocess.run(scp_cmd, shell=True, check=True)
                        print(f"Downloaded: {remote_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to fetch {remote_file}: {e.stderr}")
