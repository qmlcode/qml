"""
Code adapted from Stack Overflow answer https://stackoverflow.com/a/31464349/7146757

This class is used to process SIGTERM signal gracefully from within the neural network training.
"""

import signal

class _GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True