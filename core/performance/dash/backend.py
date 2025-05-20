import os
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv

from core.performance.performance_report import PerformanceReport
from core.performance.sync_manager import DatabaseSyncManager, ServerHandler
from core.services.backend_api_client import BackendAPIClient

load_dotenv()


class DashBackend:
    def __init__(self,
                 root_path: str,
                 sync_manager: DatabaseSyncManager):
        self.root_path = root_path
        self.sync_manager = sync_manager
        self.backend_api_clients: Dict[str, Any] = {}
        self.performance_reports: Dict[str, PerformanceReport] = {}

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

    def update_performance_reports(self):
        for name, server in self.sync_manager.servers.items():
            try:
                performance_report = PerformanceReport(
                    mongo_uri=os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/"),
                    database="quants_lab",
                    root_path=self.root_path,
                )
                performance_report.load_data(server.local_dbs, name)
                self.performance_reports[name] = performance_report
            except Exception as e:
                print(e)

    async def connect_backend_api_clients(self):
        for name, server in self.sync_manager.servers.items():
            try:
                backend_api_client = BackendAPIClient(host=server.host,
                                                      port=server.port,
                                                      username=server.backend_api_user,
                                                      password=server.backend_api_password)
                await backend_api_client.get_accounts()
                self.backend_api_clients[name] = {}
                self.backend_api_clients[name]["instance"] = backend_api_client
                self.backend_api_clients[name]["data"] = await self.get_backend_api_data(backend_api_client)
            except Exception as e:
                print(e)

    @staticmethod
    async def get_backend_api_data(backend_api_client: BackendAPIClient):
        data = {}
        bots_status = await backend_api_client.get_active_bots_status()
        if bots_status.get("status") == "success":
            data["bots_status"] = bots_status
        return data
