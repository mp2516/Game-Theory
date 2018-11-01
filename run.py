from visualisation.server import server

# model = Model(50, 10, 10)
# for i in trange(100):
#     model.step()
#
# value = model.datacollector.get_model_vars_dataframe()
# value.plot()
# plt.show()
server.port = 8521 # The default
server.launch()