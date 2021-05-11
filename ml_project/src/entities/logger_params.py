from dataclasses import dataclass


@dataclass()
class LoggerParams:
    """ Logger config ORM class. """
    path: str = 'ml_project.log'
    format: str = '%(asctime)s %(levelname)s %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    level: int = 20
    mode: str = ''
    stdout: bool = True
