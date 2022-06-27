import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Trana:
	kolumner = ['valresultat', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']

	data = [
		[25.1, np.nan, np.nan, np.nan, np.nan, np.nan, 26.7, 26.3, 25.6],  # C 1973
		[24.1, np.nan, 26.6, 25.9, 25.1, 24.7, 23.9, 23.5, 22.8],  # C 1976
		[18.1, np.nan, 23.5, 22.8, 21.8, 20.3, 20.6, 19.9, 21.1],  # C 1979
		[15.5, np.nan, np.nan, np.nan, np.nan, 18.0, 16.5, 15.6, 12.8],  # C 1982
		[10.0, np.nan, np.nan, np.nan, np.nan, np.nan, 14.7, 14.4, 12.6],  # C 1985
		[11.3, np.nan, 9.7, 10.3, 10.4, 9.9, 9.4, 9.8, 9.3],  # C 1988
		[8.5, np.nan, np.nan, 11.8, 12.2, 11.0, 11.6, 10.7, 9.8],  # C 1991
		[7.7, np.nan, np.nan, 8.1, 6.5, 6.5, 6.2, 6.6, 7.9],  # C 1994
		[5.1, 7.6, 8.3, 6.8, 6.8, 6.9, 6.8, 6.1, 5.9],  # C 1998
		[6.2, 5.0, 4.7, 4.5, 4.6, 4.6, 7.2, 5.8, 5.4],  # C 2002
		[7.9, 5.8, 6.5, 6.6, 6.6, 6.7, 6.5, 5.9, 6.2],  # C 2006
		[6.6, 7.1, 6.5, 6.2, 6.2, 5.9, 5.5, 5.0, 4.6],  # C 2010
		[6.1, 6.0, 4.7, 5.7, 5.0, 4.8, 4.2, 4.6, 4.9],  # C 2014
		[8.6, 6.2, 6.5, 6.5, 6.6, 7.3, 11.7, 9.5, 8.8],  # C 2018

		[9.4, np.nan, np.nan, np.nan, np.nan, np.nan, 13.4, 13.0, 12.5],  # FP 1973
		[11.1, np.nan, 8.7, 8.2, 8.5, 7.6, 8.9, 10.1, 10.8],  # FP 1976
		[10.6, np.nan, 11.0, 9.3, 9.6, 10.1, 12.1, 13.7, 13.6],  # FP 1979
		[5.9, np.nan, np.nan, np.nan, np.nan, 10.0, 8.7, 7.2, 6.4],  # FP 1982
		[14.2, np.nan, np.nan, np.nan, np.nan, np.nan, 6.2, 6.6, 6.5],  # FP 1985
		[12.2, np.nan, 17.9, 16.7, 16.0, 16.0, 16.0, 16.6, 15.2],  # FP 1988
		[9.1, np.nan, np.nan, 11.9, 12.1, 12.4, 12.5, 11.9, 10.0],  # FP 1991
		[7.2, np.nan, np.nan, 8.8, 6.7, 6.8, 7.2, 6.8, 6.7],  # FP 1994
		[4.7, 6.6, 7.1, 5.2, 5.9, 6.5, 5.9, 6.5, 6.8],  # FP 1998
		[13.4, 4.9, 4.2, 5.3, 5.2, 4.6, 4.9, 4.9, 5.1],  # FP 2002
		[7.5, 13.7, 14.5, 13.6, 11.7, 12.0, 11.7, 11.1, 10.9],  # FP 2006
		[7.1, 6.8, 5.9, 6.5, 6.8, 6.0, 5.5, 6.5, 5.8],  # FP 2010
		[5.4, 6.8, 6.2, 6.0, 5.3, 5.7, 6.4, 5.8, 5.1],  # FP 2014
		[5.5, 5.4, 4.6, 5.5, 5.4, 4.9, 5.0, 4.5, 4.9],  # FP 2018

		[14.3, np.nan, np.nan, np.nan, np.nan, np.nan, 10.9, 11.4, 12.8],  # M 1973
		[15.6, np.nan, 14.0, 14.1, 14.3, 15.1, 15.3, 15.3, 16.9],  # M 1976
		[20.3, np.nan, 15.0, 14.5, 14.5, 15.5, 15.6, 15.5, 16.0],  # M 1979
		[23.6, np.nan, np.nan, np.nan, np.nan, 21.0, 21.0, 20.8, 22.6],  # M 1982
		[21.3, np.nan, np.nan, np.nan, np.nan, np.nan, 25.9, 27.8, 27.6],  # M 1985
		[18.3, np.nan, 20.4, 18.8, 20.4, 18.8, 19.9, 18.9, 20.6],  # M 1988
		[21.9, np.nan, np.nan, 17.8, 20.4, 24.2, 26.2, 29.1, 23.4],  # M 1991
		[22.4, np.nan, np.nan, 22.5, 20.0, 21.4, 19.6, 19.6, 21.8],  # M 1994
		[22.9, 22.9, 25.1, 25.7, 25.7, 25.8, 30.4, 29.2, 27.2],  # M 1998
		[15.2, 24.6, 26.4, 25.2, 24.6, 23.9, 23.6, 23.8, 21.9],  # M 2002
		[26.2, 15.3, 16.8, 20.0, 21.6, 23.4, 27.7, 25.9, 25.9],  # M 2006
		[30.1, 24.9, 23.9, 22.6, 22.4, 24.8, 29.9, 26.2, 29.2],  # M 2010
		[23.3, 32.6, 31.3, 33.4, 28.4, 27.9, 27.2, 25.5, 22.9],  # M 2014
		[19.8, 24.8, 26.2, 23.6, 25.7, 23.6, 18.5, 22.0, 22.2],  # M 2018

		[1.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1973
		[1.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1976
		[1.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1979
		[1.9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1982
		[2.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1985
		[2.9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # KD 1988
		[7.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.1],  # KD 1991
		[4.1, np.nan, np.nan, 6.8, 5.0, 4.4, 4.5, 4.3, 4.4],  # KD 1994
		[11.8, 4.2, 4.0, 3.8, 3.7, 4.7, 4.6, 4.6, 4.7],  # KD 1998
		[9.1, 11.1, 11.9, 11.1, 10.9, 12.5, 10.1, 9.6, 9.2],  # KD 2002
		[6.6, 8.5, 8.2, 7.0, 6.2, 5.5, 4.4, 4.8, 5.8],  # KD 2006
		[5.6, 5.7, 4.4, 4.6, 4.6, 4.5, 4.3, 4.8, 4.5],  # KD 2010
		[4.6, 4.3, 3.7, 3.9, 3.7, 4.1, 3.5, 3.8, 3.9],  # KD 2014
		[6.3, 3.9, 3.8, 3.7, 3.2, 3.2, 3.3, 3.3, 3.0],  # KD 2018

		[43.6, np.nan, np.nan, np.nan, np.nan, np.nan, 40.6, 41.3, 41.3],  # S 1973
		[42.7, np.nan, 43.9, 44.7, 44.6, 45.1, 44.3, 44.0, 42.3],  # S 1976
		[43.2, np.nan, 43.8, 47.6, 48.1, 47.7, 45.2, 44.2, 42.0],  # S 1979
		[45.6, np.nan, np.nan, np.nan, np.nan, 43.8, 46.5, 48.9, 50.6],  # S 1982
		[44.7, np.nan, np.nan, np.nan, np.nan, np.nan, 44.4, 41.7, 43.7],  # S 1985
		[43.2, np.nan, 43.6, 46.4, 44.9, 42.8, 42.0, 43.2, 42.6],  # S 1988
		[37.6, np.nan, np.nan, 43.8, 39.3, 36.2, 32.8, 31.2, 32.1],  # S 1991
		[45.3, np.nan, np.nan, 39.1, 46.0, 45.8, 49.8, 50.8, 50.1],  # S 1994
		[36.4, 44.9, 36.0, 34.2, 36.9, 34.7, 34.9, 39.2, 40.1],  # S 1998
		[39.8, 36.3, 34.1, 36.1, 35.4, 35.3, 36.7, 39.9, 43.1],  # S 2002
		[35.0, 41.3, 38.3, 36.9, 37.7, 37.9, 34.7, 37.1, 37.5],  # S 2006
		[30.7, 40.4, 45.0, 45.9, 44.7, 42.3, 36.6, 36.5, 33.8],  # S 2010
		[31.0, 28.7, 33.9, 27.2, 37.6, 34.4, 35.1, 34.9, 35.8],  # S 2014
		[28.3, 31.9, 29.3, 27.8, 28.1, 28.4, 30.2, 31.8, 27.9],  # S 2018

		[5.3, np.nan, np.nan, np.nan, np.nan, np.nan, 5.7, 5.2, 5.2],  # V 1973
		[4.8, np.nan, 4.8, 5.0, 5.1, 5.2, 5.0, 4.8, 4.8],  # V 1976
		[5.6, np.nan, 4.6, 4.1, 4.1, 4.4, 4.6, 4.7, 4.9],  # V 1979
		[5.6, np.nan, np.nan, np.nan, np.nan, 5.7, 5.6, 5.4, 5.4],  # V 1982
		[5.4, np.nan, np.nan, np.nan, np.nan, np.nan, 4.5, 4.8, 5.0],  # V 1985
		[5.8, np.nan, 5.1, 4.3, 4.5, 4.1, 4.2, 4.4, 4.7],  # V 1988
		[4.5, np.nan, np.nan, 5.5, 6.2, 7.0, 7.8, 6.4, 5.4],  # V 1991
		[6.2, np.nan, np.nan, 4.1, 3.1, 3.4, 3.4, 3.5, 3.9],  # V 1994
		[12.0, 6.7, 12.9, 11.7, 11.6, 12.0, 10.4, 8.4, 8.5],  # V 1998
		[8.4, 12.0, 13.2, 12.2, 13.3, 14.0, 12.6, 11.4, 10.4],  # V 2002
		[5.9, 9.1, 9.1, 9.3, 9.5, 7.6, 7.0, 5.7, 5.8],  # V 2006
		[5.6, 5.4, 5.0, 5.1, 5.1, 5.7, 5.7, 5.1, 5.6],  # V 2010
		[5.7, 5.0, 4.6, 5.6, 6.0, 5.2, 6.5, 6.2, 7.6],  # V 2014
		[8.0, 5.8, 6.0, 5.7, 6.6, 7.0, 6.4, 6.8, 7.4],  # V 2018

		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1973
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1976
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1979
		[1.7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1982
		[1.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1985
		[5.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # MP 1988
		[3.4, np.nan, np.nan, 5.7, 6.2, 5.7, 4.7, 4.0, 2.6],  # MP 1991
		[5.0, np.nan, np.nan, 3.1, 2.5, 2.6, 2.9, 2.5, 2.6],  # MP 1994
		[4.5, 6.7, 6.0, 12.1, 8.7, 7.4, 6.4, 4.9, 5.7],  # MP 1998
		[4.6, 4.3, 4.1, 4.5, 4.6, 3.7, 3.4, 3.3, 3.2],  # MP 2002
		[5.2, 3.9, 4.6, 4.5, 4.6, 4.7, 4.4, 4.1, 4.7],  # MP 2006
		[7.3, 5.5, 5.5, 5.2, 5.9, 6.1, 6.0, 8.4, 10.7],  # MP 2010
		[6.9, 8.8, 8.8, 11.5, 7.9, 8.5, 8.1, 8.7, 8.3],  # MP 2014
		[4.4, 7.3, 6.7, 5.7, 4.8, 5.0, 4.5, 4.1, 4.2],  # MP 2018

		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1973
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1976
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1979
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1982
		[0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1985
		[0.02, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1988
		[0.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1991
		[0.3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1994
		[0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 1998
		[1.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 2002
		[2.9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 2006
		[5.7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # SD 2010
		[12.9, 6.3, 5.5, 5.4, 5.4, 8.1, 7.7, 9.1, 7.7],  # SD 2014
		[17.5, 12.2, 14.8, 19.6, 17.5, 17.6, 17.9, 15.2, 18.8],  # SD 2018
	]

	def trana_data(self):
		df = pd.DataFrame(data=self.data, columns=self.kolumner)
		df.head()

		# mål och prediktorer
		mal = self.kolumner[0]
		prediktorer = self.kolumner[1:]
		x = df[prediktorer].values
		y = df[mal].values
		xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

		xgbr = xgb.XGBRegressor(learning_rate=0.01, booster="gbtree", max_depth=6, n_estimators=1000, gamma=0.5, eta=0.1, subsample=0.6, objective="reg:squarederror")

		xgbr.fit(xtrain, ytrain)
		poang = xgbr.score(xtrain, ytrain)
		print("Träningspoäng: ", poang)

		scores = cross_val_score(xgbr, xtrain, ytrain, cv=10)
		print("Korsvalidering, medelpoäng: %.2f" % scores.mean())

		kfold = KFold(n_splits=10, shuffle=True)
		kd_kv = cross_val_score(xgbr, xtrain, ytrain, cv=kfold)
		print("K-delad korsvalidering, medelpoäng: %.2f" % kd_kv.mean())

		# prediktion
		ypred = xgbr.predict(xtest)
		mse = mean_squared_error(ytest, ypred)
		print("MSE: %.2f" % mse)
		print("RMSE: %.2f" % (mse ** (1/2.0)))

		x_axel = range(len(ytest))
		plt.plot(x_axel, ytest, label="Grunddata", linewidth=3.0)
		plt.plot(x_axel, ypred, label="Testfas", linewidth=3.0)
		plt.title("Test och prediktioner, XGBoost", fontsize=30)
		plt.legend(fontsize=20)
		plt.grid(True)
		plt.show()

		d_c = [8.6, 7.3, 7.4, 6.0, 7.6, 9.5, 8.4, 6.7]
		d_fp = [4.3, 3.2, 3.8, 3.3, 3.0, 2.5, 2.5, 3.4]
		d_m = [19.6, 16.9, 18.4, 20.1, 22.1, 22.4, 22.7, 21.3]
		d_kd = [5.4, 12.6, 6.8, 6.4, 5.4, 4.5, 4.6, 5.2]
		d_s = [30.3, 27.2, 25.0, 33.7, 29.4, 28.2, 29.1, 33.3]
		d_v = [7.8, 8.2, 8.6, 8.2, 9.3, 8.9, 9.2, 7.8]
		d_mp = [4.0, 5.7, 5.2, 4.1, 4.2, 3.8, 3.9, 3.3]
		d_sd = [18.6, 17.4, 22.7, 17.1, 17.6, 18.9, 18.6, 17.0]
		d = [d_c, d_fp, d_m, d_kd, d_s, d_v, d_mp, d_sd]

		prediktion = np.round(xgbr.predict(d), decimals=2)
		print(prediktion)
		print(sum(prediktion))


Trana().trana_data()
