import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.model_selection import train_test_split

def main():
    st.title("Student's Percentage Prediction Model ")
    st.markdown('By Shallum Israel')
    url = "http://bit.ly/w-data"

    data = pd.read_csv(url)
    if st.checkbox('Show Data'):
        st.table(data)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    st.subheader('Regression plot')
    line = regressor.coef_ * X + regressor.intercept_
    plt.title('Regression Line')
    plt.scatter(X, y)
    plt.plot(X, line)
    st.pyplot()

    st.subheader('Data plot')
    data.plot(x='Hours', y='Scores', style='o')
    plt.title('Hours vs Percentage')
    plt.xlabel('Hours Studied')
    plt.ylabel('Percentage Score')
    st.pyplot()

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.subheader('Expected vs Predicted')
    st.table(df)

    st.subheader("Predict your Score")
    hours = st.number_input('No of hours studied')
    scoreEx = st.number_input('Score Expectation')
    if st.button("Predict your score"):
        own_pred = regressor.predict([[hours]])
        st.write("No of Hours = {}".format(hours))
        st.write("Score Expected = {}".format(scoreEx))
        st.write("Predicted Score = {}".format(own_pred[0]))

    st.subheader('Evaluation metrics')
    from sklearn import metrics
    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    st.write('Accuracy', regressor.score(X, y))


if __name__ == '__main__':
     main()
