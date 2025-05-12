# backend.py
import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from core.services.backend_api_client import BackendAPIClient

load_dotenv()


async def fetch_servers_info():
    backend_api_servers_str = os.getenv("BACKEND_API_SERVERS", '{"master": "localhost"}')
    backend_api_servers = json.loads(backend_api_servers_str)

    backend_api_info = []
    for backend_api_name, backend_api_host in backend_api_servers.items():
        try:
            backend_api_client = BackendAPIClient(backend_api_host)
            archived_dbs = await backend_api_client.list_databases()
            running_bots = await backend_api_client.get_active_bots_status()
            info_dict = {
                "name": backend_api_name,
                "host": backend_api_host,
                "archived_databases": len(archived_dbs),
                "running_bots": len(running_bots["data"]),
                "status": "Connected"
            }
        except Exception as e:
            info_dict = {
                "name": backend_api_name,
                "host": backend_api_host,
                "archived_databases": 0,
                "running_bots": 0,
                "status": "Disconnected"
            }
        backend_api_info.append(info_dict)

    return pd.DataFrame(backend_api_info)


