from game_theory.config import Config

file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

if model_config.parameters['simulation']:
    from game_theory.server import server
    server.port = 8521  # The default
    server.launch()
else:
    from game_theory.analysis import RPSAnalysis
    RPSAnalysis.run_model(model_config.parameters, model_config.batchrunning)