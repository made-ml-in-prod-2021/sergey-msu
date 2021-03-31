from dataclasses import dataclass


@dataclass()
class EDAReportParams:
    """ EDA report config ORM class. """
    name: str = 'etl_report'
    data_path: str = 'heart.csv'
    save_path: str = 'report.txt'
