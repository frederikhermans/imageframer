def _create_logger(name):
    import logging
    import sys

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: '
                                  '%(message)s', datefmt='%M:%H:%S')
    ch.setFormatter(formatter)

    log.addHandler(ch)
    return log


log = _create_logger('imageframer')
