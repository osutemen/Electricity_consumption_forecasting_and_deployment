# Electricity Consumption Forecasting and Deployment

This project has been completed as the final project of VBO's 3rd MLOps bootcamp. A time-series model was built using real-time energy 
consumption data acquired by EPİAŞ. Based on this model, predictions are made for a given day and hour.

In this project, predictions are made using the ML model that was created. The prediction results were stored to a MySQL database for 
data drift analysis. Mlflow was used to monitor the hyperparameters and evaluation metrics of the model. The model was queried 
through FastAPI by providing the day and hour inputs for making predictions, and FastAPI was also used to detect data drift in the model.

These were done using Docker Compose images.

