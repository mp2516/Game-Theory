import json

from .logger import logger


class Config(object):
    """
    General configuration object for application. Loads a config file as a JSON.
    Contains umbrella config settings.
    """

    APP_NAME = "GAME THEORY SIMULATOR"
    VERSION = "0.0.1"

    def __init__(self, data):
        """
        Args:
            data: Load file data directly into config object.
        """
        self.__dict__ = json.loads(data)
        logger.critical("Loaded config successfully".format())
