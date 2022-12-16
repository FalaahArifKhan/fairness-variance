import logging


class CustomHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        fmt = '%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s'
        fmt_date = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)
