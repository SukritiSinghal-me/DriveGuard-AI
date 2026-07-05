import time

from utils.alarm import start_alarm
from utils.alarm import stop_alarm

print("Alarm Starts")

start_alarm()

time.sleep(5)

stop_alarm()

print("Alarm Stopped")