from lqsvg.testing.fixture import standard_fixture

n_state = standard_fixture((2, 3), "NState")
n_ctrl = standard_fixture((2, 3), "NCtrl")
horizon = standard_fixture((1, 3, 10), "Horizon")
seed = standard_fixture((1, 2, 3), "Seed")
