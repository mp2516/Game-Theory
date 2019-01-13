import datetime
import logging.config
from .config import Config
import logaugment

file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

LEVEL = model_config.logger_level
# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,  # this fixes the problem
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': LEVEL,
            'propagate': True
        }
    }
})

formatter = logging.Formatter("%(time_since_last)s: %(message)s")
handler.setFormatter(formatter)

NO_COLOR = "\33[m"
# noinspection SpellCheckingInspection
RED, GREEN, ORANGE, BLUE, PURPLE, LBLUE, GREY = \
    map("\33[%dm".__mod__, range(31, 38))


def add_color(logger_method, _color):
    # noinspection PyShadowingNames
    def wrapper(message, *args, **kwargs):
        color = kwargs.pop("color", _color)
        if isinstance(color, int):
            color = "\33[%dm" % color
        return logger_method(
            # the coloring is applied here.
            color + message + NO_COLOR,
            *args, **kwargs
        )

    return wrapper


for level, color in zip((
        "info", "warn", "error", "debug"), (
        45, GREY, RED, ORANGE
)):
    setattr(logger, level, add_color(getattr(logger, level), color))


# logger.addHandler(handler)


# noinspection PyUnusedLocal
def process_record(record):
    """

    :param record: Previous log.
    :return: Time since last log.
    """
    now = datetime.datetime.utcnow()
    try:
        delta = now - process_record.now
    except AttributeError:
        delta = 0
    process_record.now = now
    return {'time_since_last': delta}


logaugment.add(logger, process_record)