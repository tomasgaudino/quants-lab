import yaml
import os
import subprocess
from typing import Any, Dict, List


class ServerHandler:
    def __init__(self, server_name: str, root_path: str, config: Dict[str, Any]):
        self.server_name = server_name
        self.user = config.get("user", "root")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8000)
        self.backend_api_user = config.get("backend_api_user", "admin")
        self.backend_api_password = config.get("backend_api_password", "admin")
        self.remote_data_path = config["data_path"]
        self.root_path = root_path

        self.local_path = os.path.join(root_path, "data/live_bot_databases", server_name)
        os.makedirs(self.local_path, exist_ok=True)

        # State variables
        self.remote_db_paths: List[str] = []
        self.remote_db_names: List[str] = []

        self.local_dbs: List[str] = []

        self.missing_dbs_paths: List[str] = []
        self.missing_dbs_names: List[str] = []

    def update_state(self):
        self.remote_db_paths = self._get_all_remote_dbs()
        self.remote_db_names = [os.path.basename(p) for p in self.remote_db_paths]
        self.local_dbs = self._get_all_local_dbs()
        self._calculate_missing_dbs()

    def fetch_missing_dbs(self):
        for remote_path in self.missing_dbs_paths:
            self._scp_file(remote_path, self.local_path)

    def _calculate_missing_dbs(self):
        local_set = set(self.local_dbs)

        self.missing_dbs_paths = [
            path for path in self.remote_db_paths
            if os.path.basename(path) not in local_set
        ]
        self.missing_dbs_names = [os.path.basename(p) for p in self.missing_dbs_paths]

    def _get_all_remote_dbs(self) -> List[str]:
        """
        List all *.sqlite files under remote_data_path/**/*. Only filter out specific unwanted names.
        """
        cmd = (
            f'ssh {self.user}@{self.host} '
            f'"find {self.remote_data_path} -type f -name \'*.sqlite\'"'
        )
        raw_paths = self._run_ssh_command(cmd)

        # Local filtering
        return [
            p for p in raw_paths
            if "/data/" in p and not p.endswith("v2_with_controllers.sqlite")
        ]

    def _get_all_local_dbs(self) -> List[str]:
        return [
            f.strip() for f in os.listdir(self.local_path)
            if f.endswith(".sqlite") and os.path.isfile(os.path.join(self.local_path, f))
        ]

    def _scp_file(self, remote_path: str, local_dir: str):
        cmd = f"scp {self.user}@{self.host}:{remote_path} {local_dir}/"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"[{self.server_name}] Downloaded: {remote_path}")
        except subprocess.CalledProcessError as e:
            print(f"[{self.server_name}] Failed to fetch {remote_path}: {e.stderr}")

    def _run_ssh_command(self, cmd: str) -> List[str]:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        except subprocess.CalledProcessError as e:
            print(f"[{self.server_name}] SSH command failed: {e.stderr}")
            return []

    def get_missing_db_names(self) -> List[str]:
        return self.missing_dbs_names

    def get_missing_db_paths(self) -> List[str]:
        return self.missing_dbs_paths


class DatabaseSyncManager:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.config_file = os.path.join(root_path, "config/remote_servers.yml")
        self.servers: Dict[str, ServerHandler] = {}
        self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r') as f:
            server_configs = yaml.safe_load(f)
        for name, cfg in server_configs.items():
            self.servers[name] = ServerHandler(name, self.root_path, cfg)

    def update_all(self):
        for server in self.servers.values():
            server.update_state()

    def fetch_all_missing(self, server_name: str) -> List[str]:
        for server in self.servers.values():
            server.fetch_missing_dbs()

    def get_all_missing_dbs(self) -> Dict[str, Dict[str, List[str]]]:
        """Returns: { server_name: { instance_name: [dbs...] } }"""
        return {
            server_name: handler.missing_dbs_names
            for server_name, handler in self.servers.items()
        }
