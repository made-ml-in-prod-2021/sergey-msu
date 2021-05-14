from logging import Logger


class NopeLogger(Logger):
    """ Nope logger for stub testing. Does nothing. """
    def info(self, msg, *args, **kwargs):
        return

    def debug(self, msg, *args, **kwargs):
        return

    def warning(self, msg, *args, **kwargs):
        return

    def warn(self, msg, *args, **kwargs):
        return

    def error(self, msg, *args, **kwargs):
        return
