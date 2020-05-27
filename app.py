import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#used to visualize the metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#metrics to be displayed
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? ")
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("mushrooms.csv")
#since data is categorical it has to be encoded since models accept only numerical vals
        labelencoder=LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

#splitting the dataset
    @st.cache(persist=True)
    def split(df):
        #target labels
        y = df.type
        #every feature except target label
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test


    #only one model can be selected at a time so the model var has the selected model
    def plot_metrics(metrics_list):
        if ('Confusion Matrix' in metrics_list):
            st.subheader("Confusion Matrix")
            #to display target labels class_names list is used
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if ('ROC Curve' in metrics_list):
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if ('Precision-Recall Curve' in metrics_list):
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ['edible', 'poisonous']#target labels:for confusion matrix
    x_train, x_test, y_train, y_test = split(df)

#selecting the classifier only when the classifier is selected
#the corresponding hyperparameters are displayed

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if (classifier == 'Support Vector Machine (SVM)'):
        st.sidebar.subheader("Model Hyperparameters")
        #choose hyperparameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        #if block entered only when button classify is clicked
        if (st.sidebar.button("Classify", key='classify')):
            st.subheader("Support Vector Machine (SVM) Results")
            #creating model from user inputs
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            #training
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            #predicting the output
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if (classifier == 'Logistic Regression'):
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if (st.sidebar.button("Classify", key='classify')):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if (classifier == 'Random Forest'):
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if (st.sidebar.button("Classify", key='classify')):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if (st.sidebar.checkbox("Show raw data", False)):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
        st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
        "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
        "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

if __name__ == '__main__':
    main()


