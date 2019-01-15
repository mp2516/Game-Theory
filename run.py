from game_theory.config import Config

file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

if __name__ == '__main__':
    if model_config.simulation:
        from game_theory.server import server
        server.port = 8521  # The default
        server.launch()
    elif model_config.rate_equations:
        # from rate_equations.basic_rate_equations import rate_equation
        from rate_equations.solve_rates import rate_solver
        # rate_equation()
        rate_solver(0.01)
    elif model_config.network_interaction_scale:
        from rate_equations.network_interaction_scale import interaction_scale
        interaction_scale()
    else:
        from game_theory.analysis import BatchRunner
        analysis = BatchRunner(model_config.batchrunning, model_config.parameters)
        BatchRunner.run_model(analysis)
