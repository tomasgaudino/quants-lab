from typing import Dict

import pandas as pd
from dotenv import load_dotenv

from core.performance.sync_manager import DatabaseSyncManager, ServerHandler
from core.services.backend_api_client import BackendAPIClient

load_dotenv()


class DashBackend:
    def __init__(self, sync_manager: DatabaseSyncManager):
        self.sync_manager = sync_manager

    async def fetch_servers_info(self):
        servers: Dict[str, ServerHandler] = self.sync_manager.servers
        servers_info = []
        for name, server in servers.items():
            try:
                backend_api_client = BackendAPIClient(server.host)
                running_bots = await backend_api_client.get_active_bots_status()
                local_dbs = server.local_dbs
                server_dbs = server.remote_db_names
                missing_dbs = server.missing_dbs_names
                info_dict = {
                    "name": name,
                    "host": server.host,
                    "local_databases": len(local_dbs),
                    "server_dbs": len(server_dbs),
                    "missing_dbs": len(missing_dbs),
                    "running_bots": len(running_bots["data"]),
                    "status": "Connected"
                }
            except Exception as e:
                info_dict = {
                    "name": name,
                    "host": server.host,
                    "local_databases": 0,
                    "server_dbs": 0,
                    "missing_dbs": 0,
                    "running_bots": 0,
                    "status": "Disconnected"
                }
            servers_info.append(info_dict)

        return pd.DataFrame(servers_info)
