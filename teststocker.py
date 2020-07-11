from stocker import Stocker

hp = Stocker("HPQ")
stock_history = hp.stock
print(stock_history.head())


model, model_data = hp.create_prophet_model()
model.plot_components(model_data)
plt.show()
