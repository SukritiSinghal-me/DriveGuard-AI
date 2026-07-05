import threading
import winsound

alarm_running = False


def play_alarm():

    global alarm_running

    while alarm_running:
        winsound.Beep(2000, 500)


def start_alarm():

    global alarm_running

    if not alarm_running:

        alarm_running = True

        threading.Thread(
            target=play_alarm,
            daemon=True
        ).start()


def stop_alarm():

    global alarm_running

    alarm_running = False