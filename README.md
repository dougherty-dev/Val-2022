# Val-2022

Prediktion av valutgången 2022 med XGBoost givet SCB:s valundersökningar 1972–2022 samt mellanliggande val.

![scb-val](https://user-images.githubusercontent.com/71740645/175857884-fdbe8ad8-1c22-43db-b7c6-bb15dd5f401f.png)

Vald algoritm är XGBRegressor med målfunktion reg:squarederror. Ett typiskt resultat:

Träningspoäng:  0.9949925656989446<br/>
Korsvalidering, medelpoäng: 0.87<br/>
K-delad korsvalidering, medelpoäng: 0.91<br/>
MSE: 6.19<br/>
RMSE: 2.49<br/>
[7.63, 6.32, 17.86, 6.9, 29.04, 8.74, 5.05, 17.79]<br/>
99.330002784729

![xboost](https://user-images.githubusercontent.com/71740645/175857950-46dfbe47-229e-455a-9422-b62d024767d0.png)

Vid upprepad körning erhölls följande resultat i medel: C: 8.2 % FP: 5.8 % M: 18.2 % KD: 6.2 % S: 28.0 % V: 9.4 % MP: 5.3 % SD: 17.0 %. Summa: 98.2 %.

![ai-prognos-valet-2022](https://user-images.githubusercontent.com/71740645/175857970-07094c7e-d382-46cf-aafc-93b2dc6c907d.png)
