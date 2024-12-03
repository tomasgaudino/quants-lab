import asyncio
import json
import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import plotly.express as px
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib import colors
import asyncpg
import pandas as pd
import numpy as np

from core.services.backend_api_client import BackendAPIClient
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

load_dotenv()
logging.basicConfig(level=logging.INFO)


# Base class for common functionalities like database connection and email sending
class TaskBase(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.name = name
        self.frequency = frequency
        self.config = config
        self.export = self.config.get("export", False)
        self.ts_client = TimescaleClient(host=self.config.get("host", "localhost"))
        self.backend_api_client = BackendAPIClient(host=self.config.get("backend_api_host", "localhost"))

    async def execute_query(self, query: str):
        """Executes a query and returns the result."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query)

    def create_email(self, subject: str, sender_email: str, recipients: List[str], body: str) -> MIMEMultipart:
        """Creates a basic email structure."""
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        message.attach(MIMEText(body, "html"))
        return message

    def add_attachment(self, message: MIMEMultipart, path: str, table: pd.DataFrame()):
        """Attaches a file to the email."""
        real_path = path + '.csv'
        table.to_csv(real_path)
        with open(real_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(path)}")
        message.attach(part)
        # Generate the heatmap PDF

    def send_email(self, message: MIMEMultipart, sender_email: str, app_password: str, smtp_server="smtp.gmail.com",
                   smtp_port=587):
        """Sends an email using the specified SMTP server."""
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.sendmail(sender_email, message["To"].split(", "), message.as_string())
            logging.info("Email sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")


class ReportGeneratorTask(TaskBase):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.csv_dict = None
        self.base_metrics = None

    async def get_base_tables(self):
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        return table_names

    @staticmethod
    def is_new(row):
        to_date = pd.to_datetime(row['to_timestamp']).date()
        from_date = pd.to_datetime(row['from_timestamp']).date()

        # Get today's date and yesterday's date
        today_date = datetime.now().date()
        yesterday_date = (datetime.now() - timedelta(days=1)).date()

        # Compare the dates
        last_data_today = to_date >= today_date
        last_data_yesterday = to_date >= yesterday_date
        first_data_yesterday = from_date >= yesterday_date
        if last_data_today:
            when = 'today'
        elif last_data_yesterday:
            when = 'yesterday'
        else:
            when = None

        # row['is_new'] = last_data_today & first_data_yesterday
        # row['when'] = when
        return (last_data_today & first_data_yesterday), when

    async def set_base_metrics(self):
        base_metrics = await self.ts_client.get_db_status_df()
        base_metrics["table_names"] = base_metrics.apply(lambda x: self.ts_client.get_trades_table_name(x["connector_name"], x["trading_pair"]), axis=1)
        # base_metrics['when'] = None
        # base_metrics['is_new'] = False
        # Apply the is_new function and unpack the tuple into 'is_new' and 'when' columns
        base_metrics[['is_new', 'when']] = base_metrics.apply(lambda x: self.is_new(x), axis=1, result_type='expand')
        self.base_metrics = base_metrics.dropna(subset=["when"]).copy()

    async def generate_heatmap(self):
        # Load the all_daily_metrics CSV into a DataFrame
        base_metrics = self.base_metrics
        # Calculate total trade amounts by trading pair and percentage
        total_trade_amounts = base_metrics.groupby('trading_pair')['trade_amount'].transform('sum')
        base_metrics['trade_amount_pct'] = (base_metrics['trade_amount'] / total_trade_amounts) * 100

        # Pivot data for heatmap-ready format with trade_amount as percentage
        heatmap_data = base_metrics.pivot(index='trading_pair', columns='day', values='trade_amount_pct')

        # Create the heatmap using Plotly
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale="Reds",
            labels={'color': 'Trade Amount (%)'},
            title="Trade Amount Heatmap by Date and Trading Pair"
        )

        # Customize the layout for readability
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Trading Pair",
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_tickfont_size=12,
            yaxis_tickfont_size=12
        )
        pdf_filename = "trade_amount_heatmap.pdf"
        # Save to PDF if required
        fig.write_image(pdf_filename)

        # Show interactive heatmap in a notebook or webpage
        fig.show()

        return pdf_filename
    @staticmethod
    def generate_heatmap_table(data: dict) -> str:
        """
        Generates an HTML table with heatmap styling for specific columns.
        """
        # Flatten the nested dictionary into a pandas DataFrame
        rows = []
        for bot, bot_data in data.items():
            for id_key, metrics in bot_data.items():
                row = {"bot_name": bot, "id_key": id_key}
                row.update(metrics["performance"])
                rows.append(row)

        df = pd.DataFrame(rows)

        # Columns to apply heatmap
        heatmap_columns = [
            "realized_pnl_quote", "unrealized_pnl_quote", "unrealized_pnl_pct",
            "realized_pnl_pct", "global_pnl_quote", "global_pnl_pct",
            "volume_traded", "open_order_volume", "inventory_imbalance"
        ]

        # Normalize the values for the heatmap
        norm = {}
        for col in heatmap_columns:
            if col in df.columns:
                norm[col] = colors.Normalize(vmin=df[col].min(), vmax=df[col].max())

        # Generate the HTML table with inline styles
        table_html = '<table border="1" style="border-collapse: collapse; width: 100%;">'
        table_html += "<thead>"
        table_html += "<tr style='background-color: #f2f2f2; font-weight: bold;'>"
        table_html += "".join([f"<th>{col}</th>" for col in df.columns])
        table_html += "</tr>"
        table_html += "</thead>"
        table_html += "<tbody>"

        for _, row in df.iterrows():
            table_html += "<tr>"
            for col in df.columns:
                value = row[col]
                if col in heatmap_columns:
                    if pd.notna(value):  # Apply heatmap only for non-NaN values
                        # Map the value to a color scale (green for positive, red for negative)
                        cmap = colors.LinearSegmentedColormap.from_list("heatmap", ["#FF6666", "#FFFFFF", "#66FF66"])
                        color = colors.to_hex(cmap(norm[col](value)))
                        table_html += f"<td style='background-color: {color}; text-align: center;'>{value:.1f}</td>"
                    else:
                        table_html += "<td style='text-align: center;'>-</td>"
                else:
                    table_html += f"<td style='text-align: center;'>{value}</td>"
            table_html += "</tr>"

        table_html += "</tbody>"
        table_html += "</table>"

        return table_html

    async def execute(self):
        print(f"\n\nDatetime:{datetime.now()}\n\n")
        await self.ts_client.connect()
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        await self.set_base_metrics()

        active_bots_status = await self.backend_api_client.get_active_bots_status()
        performance_data = active_bots_status["data"]
        performance_by_bot_dict = {instance_name: data.get("performance") for instance_name, data in performance_data.items()}
        # performance_string = json.dumps(performance_by_bot_dict, indent=4)
        # Generate the report and prepare the email
        report = self.generate_report(table_names, performance_by_bot_dict)

        message = self.create_email(
            subject="Database Refresh Report - Thinking Science Journal",
            sender_email=self.config["email"],
            recipients=self.config["recipients"],
            body=report
        )

        # Attach CSV files, heatmap PDF, and other files
        # for filename in ["all_daily_metrics.csv", heatmap_pdf] + [f"{k}.csv" for k in csv_dict]:
        # TODO: try to replace all_daily_metrics by a class attribute like self.daily_metrics
        for filename, table in self.csv_dict.items():
            try:
                self.add_attachment(message, filename + '.csv', table)
            except Exception as e:
                print(f"Unable to attach file {filename}: {e}")

        # Send the email
        self.send_email(message, sender_email=self.config["email"], app_password=self.config["email_password"])

    def generate_report(self, table_names: List[str], bots_report: Dict) -> (str, Dict[str, pd.DataFrame]):
        # Analyze the database for missing pairs, outdated pairs, etc.
        final_df = self.base_metrics
        missing_pairs_list = [pair for pair in table_names if pair not in final_df['table_names'].unique()]
        outdated_pairs_list = [pair for pair in table_names if
                               pair in final_df[final_df['when'] == 'yesterday']['table_names'].unique()]
        correct_pairs_list = [pair for pair in table_names if
                              pair in final_df[final_df['when'] == 'today']['table_names'].unique()]
        new_pairs_list = [pair for pair in table_names if
                          pair in final_df[final_df['is_new']]['table_names'].unique()]

        # Start building the email body
        report = f"""
        <html>
            <body>
                <p>Hello Mr. Pickantell!:</p>
                <p><b>Here are your bots running:</b></p>
        """

        # Generate the heatmap table for the bots report
        if bots_report:
            html_table = self.generate_heatmap_table(bots_report)
            report += html_table
        else:
            report += "<p>You have no running bots. Please check your system.</p>"

        # Add database analysis details
        report += f"<p><b>Here's a quick review of your database:</b></p>"

        # Create a more readable format for the lists
        report += f"<ul><li><b>Missing trading pairs (not updated in 2 days):</b> {len(missing_pairs_list)} pairs</li>"
        if missing_pairs_list:
            report += "<ul>" + "".join(f"<li>{pair}</li>" for pair in missing_pairs_list) + "</ul>"
        report += f"<li><b>Outdated trading pairs (not updated since yesterday):</b> {len(outdated_pairs_list)} pairs</li>"
        if outdated_pairs_list:
            report += "<ul>" + "".join(f"<li>{pair}</li>" for pair in outdated_pairs_list) + "</ul>"
        report += f"<li><b>Correct trading pairs (up to date):</b> {len(correct_pairs_list)} pairs</li>"
        if correct_pairs_list:
            report += "<ul>" + "".join(f"<li>{pair}</li>" for pair in correct_pairs_list) + "</ul>"
        report += f"<li><b>New trading pairs:</b> {len(new_pairs_list)} pairs</li>"
        if new_pairs_list:
            report += "<ul>" + "".join(f"<li>{pair}</li>" for pair in new_pairs_list) + "</ul>"

        # Bold the "Additional Database Flux Information" section
        report += f"<p><b>Additional Database Flux Information:</b></p>"
        report += f"<p>--> Amount of trading pairs missing (no info for 2 days) out of total pairs: {(len(missing_pairs_list) / len(table_names) * 100):.2f}%</p>"
        report += f"<p>--> Outdated pairs (no info since yesterday) out of total pairs: {(len(outdated_pairs_list) / len(table_names) * 100):.2f}%</p>"
        report += f"<p>--> Correct pairs (updated info) out of total pairs: {(len(correct_pairs_list) / len(table_names) * 100):.2f}%</p>"
        report += f"<p>--> New pairs out of total pairs:: {(len(new_pairs_list) / len(table_names) * 100):.2f}%</p>"

        report += f"<p><b>For more information visit the attached files:</b></p>"
        report += f"<p>++ all_daily_metrics.csv: general information about current databases</p>"
        report += f"<p>++ trade_amount_heatmap.pdf: Per trading pair - % of total trades downloaded per day</p>"
        report += f"<p>See you soon and don't forget to be awesome!!</p>"

        # Prepare CSV data for additional attachments
        csv_dict = {
            "missing_pairs": pd.Series(missing_pairs_list),
            "outdated_pairs": pd.Series(outdated_pairs_list),
            "correct_pairs": pd.Series(correct_pairs_list),
            "new_pairs": pd.Series(new_pairs_list),
        }
        self.csv_dict = {key: df for key, df in csv_dict.items() if len(df) > 20}
        self.csv_dict['all_daily_metrics'] = self.base_metrics

        return report


async def main():
    config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "backend_api_host": os.getenv("TRADING_HOST", "localhost"),
        "email": "thinkingscience.ts@gmail.com",
        "email_password": "dqtn zjkf aumv esak",
        "recipients": ["palmiscianoblas@gmail.com", "federico.cardoso.e@gmail.com", "apelsantiago@gmail.com",  "tomasgaudino8@gmail.com"],
        # "recipients": ["palmiscianoblas@gmail.com"],
        "export": True
    }
    task = ReportGeneratorTask(name="Report Generator", frequency=timedelta(hours=12), config=config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())


# <li>Missing trading pairs (not updated in 2 days): {len(missing_pairs_list)} pairs</li>
#                     <li>{missing_pairs_list}</li>
#                     <li>Outdated trading pairs (not updated since yesterday): {len(outdated_pairs_list)} pairs</li>
#                     <li>{outdated_pairs_list}</li>
#                     <li>Correct trading pairs (up to date): {len(correct_pairs_list)} pairs</li>
#                     <li>{correct_pairs_list}</li>
#                     <li>New trading pairs: {len(new_pairs_list)} pairs</li>
#                     <li>{new_pairs_list}</li>