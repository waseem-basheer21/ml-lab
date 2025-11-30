from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

data = fetch_california_housing()

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(
    X,y ,test_size=0.3,random_state=42
)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MSE:",mse)
print("r2 Score:",r2)
print("Actual Price:",y_train[:10])
print("Predicted Price:",y_pred[:10])