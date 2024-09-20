import subprocess
import psutil
import os
import signal
from crontab import CronTab


class PyScriptController:
    def __init__(self, script_path):
        """
        Initialize with the path to the script.
        """
        self.script_path = script_path
        self.process = None
        self.cron = CronTab(user=True)

    def is_running(self):
        """
        Check if the script is running and print status.
        """
        a = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                a.append(proc)
                # Join the command line and check if script path is present
                cmdline = ' '.join(proc.info['cmdline'])
                if self.script_path in cmdline:
                    print(f"Script {self.script_path} is running with PID {proc.info['pid']}. Command line: {cmdline}")
                    return proc.info['pid']
            except Exception as e:
                print(f"Error reading process info: {e}")
        print(f"Script {self.script_path} is not running.")
        return None

    def trigger(self):
        """
        Trigger the script and run it in a subprocess.
        """
        if self.is_running():
            return

        print(f"Starting script {self.script_path}...")
        self.process = subprocess.run(
            args=f'nohup python3 {self.script_path} > /dev/null 2>&1 &',
            shell=True, text=True)
        pid = self.is_running()
        if pid:
            print(f"Script {self.script_path} started with PID {pid}.")
        else:
            print(f"Error starting script: {self.process.stderr}")

    def stop(self):
        """
        Stop the script if it is running.
        """
        pid = self.is_running()
        if pid:
            print(f"Stopping script with PID {pid}...")
            os.kill(pid, signal.SIGTERM)
            print(f"Script {self.script_path} stopped.")
        else:
            print(f"No running script found for {self.script_path}.")

    def logs(self):
        """
        Print the logs of the script.
        """
        try:
            with open(f"{self.script_path[:-3]}.log", "r") as f:
                print(f.read())
        except FileNotFoundError:
            print(f"No log file found for {self.script_path}.")

    # Cron Management Methods

    def add_cron(self, minute='0', hour='*', day_of_month='*', month='*', day_of_week='*'):
        """
        Add a new cron job to run the script at the specified schedule using human-readable arguments.
        Example: add_cron(minute=0, hour=3) -> Runs at 3:00 AM every day.
        """
        if self.find_cron():
            print(f"Cron job for {self.script_path} already exists.")
            return

        schedule = self._build_schedule(minute, hour, day_of_month, month, day_of_week)
        job = self.cron.new(command=f'python3 {self.script_path}')
        job.setall(schedule)
        self.cron.write()
        print(f"Cron job added for {self.script_path} with schedule: {schedule}.")

    def find_cron(self):
        """
        Find and return the cron job for the script, if it exists.
        """
        for job in self.cron:
            if self.script_path in job.command:
                return job
        return None

    def remove_cron(self):
        """
        Remove the cron job for the script if it exists.
        """
        job = self.find_cron()
        if job:
            self.cron.remove(job)
            self.cron.write()
            print(f"Cron job for {self.script_path} removed.")
        else:
            print(f"No cron job found for {self.script_path}.")

    def list_cron(self):
        """
        List all cron jobs for the current user.
        """
        for job in self.cron:
            print(f"Cron job: {job}")

    def modify_cron(self, minute='0', hour='*', day_of_month='*', month='*', day_of_week='*'):
        """
        Modify the cron job schedule for the script if it exists using human-readable arguments.
        Example: modify_cron(minute=30, hour=14) -> Runs at 2:30 PM every day.
        """
        job = self.find_cron()
        if job:
            schedule = self._build_schedule(minute, hour, day_of_month, month, day_of_week)
            job.setall(schedule)
            self.cron.write()
            print(f"Cron job for {self.script_path} updated to schedule: {schedule}.")
        else:
            print(f"No cron job found for {self.script_path}.")

    @staticmethod
    def _build_schedule(minute='*', hour='*', day_of_month='*', month='*', day_of_week='*'):
        """
        Build the cron schedule string from individual parts.
        Default values are wildcards ('*'), meaning every unit of time.
        """
        return f"{minute} {hour} {day_of_month} {month} {day_of_week}"