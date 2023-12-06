from aif360.algorithms.postprocessing import EqOddsPostprocessing

class CustomEqOddsPostprocessing(EqOddsPostprocessing):
    def __init__(self, unprivileged_groups, privileged_groups, seed=None):
        super().__init__(unprivileged_groups, privileged_groups, seed)
        
        self.saved_params = []