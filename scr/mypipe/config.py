from dataclasses import dataclass


@dataclass
class Config:
    EXP_NAME: str = ""
    RUN_NAME: str = ""
    TARGET: str = ""
    INPUT: str = '../input'
    OUTPUT: str = '../output'
    SUBMISSION: str = '../submission'
    NOTEBOOKS: str = '../notebooks'

    def __post_init__(self):
        self.EXP = self.OUTPUT + f'/{self.EXP_NAME}'
        self.PREDS: str = self.EXP + '/preds'
        self.COLS: str = self.EXP + '/cols'
        self.TRAINED: str = self.EXP + '/trained'
        self.FEATURE: str = self.EXP + '/feature'
        self.REPORTS: str = self.EXP + '/reports'
