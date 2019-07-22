import pickle
import matplotlib
# test different image backends for matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
	try:
		matplotlib.use(gui,warn=False, force=True)
		from matplotlib import pyplot as plt
		break
	except:
		continue
import seaborn as sns

from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from depablo_box.utils import NA_encoder,calculate_descriptor

class model:
	def __init__(self, df, input_properties, output_property, na_strategy="mean"):
		self.df = df
		self.input_properties = input_properties
		self.output_property = output_property
		self.na_strategy = na_strategy

		# will be filled during train
		self.trained_model = None
		self.r_2 = None

	def feature_importance(self):
		"""
		Plot histogram of feature importances for predicting target
		"""

		imp,properties = zip(*sorted(zip(self.trained_model.coef_[0],self.input_properties)))
		plt.barh(range(len(properties)), imp, align='center')
		plt.yticks(range(len(properties)), properties)
		plt.tight_layout()
		plt.show()

	def train(self, model_type):
		"""
		Train a regression model based on input_properties to predict output_property

		Parameters
		----------------------
		model_type: String
			model algorithm to train data on 

		"""
		algorithms = {
			"Support Vector Regression": svm.SVR(kernel="linear"),
			"Linear Regression": linear_model.LinearRegression(),
			"Ridge Regression": linear_model.Ridge(),
			"Lasso Regression": linear_model.Lasso()
		}

		encoder = NA_encoder(numerical_strategy=self.na_strategy)
		self.df = NA_encoder().fit_transform(self.df)
		X = self.df[self.input_properties]
		y = self.df[self.output_property]

		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


		regressor = algorithms[model_type]
		regressor.fit(X_train, y_train)

		self.r_2 = regressor.score(X_test, y_test)
		self.trained_model = regressor

	def predict(self, input_properties):
		"""
		Parameters
		-----------------------------
		input_properties: [[Any], [Any], [Any]]
			List of arrays where each array represents a row of input 

		Return 
		-----------------------------
		predicted_property: [value]
			Array of predicted values, each element represents the predicted value for the corresponding row in the input matrix.

		"""

		return self.trained_model.predict(input_properties)

	def export_fitted_model(self,outpath):
		"""
		Export fitted model as pickle file to outpath

		Parameters
		-----------------------------
		outpath: String
			file path to write fitted model pickle to

		"""

		with open(outpath, 'wb') as f:
			pickle.dump(self.trained_model,f)

