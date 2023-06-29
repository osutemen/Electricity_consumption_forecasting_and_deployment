# Electricity Consumption Forecasting and Deployment

This project has been completed as the final project of VBO's 3rd MLOps bootcamp. A time-series model was built using real-time energy consumption data acquired by EPİAŞ. Based on this model, forecasting are made for a given day and hour.

In this project, predictions are made using the ML model that was created. The prediction results were stored to a MySQL database for data drift analysis. Mlflow was used to monitor the hyperparameters and evaluation metrics of the model. The model was queried through FastAPI by providing the day and hour inputs for making predictions, and FastAPI was also used to detect data drift in the model.

These were done using Docker compose containers.

P.S.The project was completed in 10 days, so the model was kept at a basic level. More accurate results can be obtained by adjusting the model. In addition, the model was automated by using Ansible in Jenkins the project and by providing CI/CD on Guvicorn. However, this part is not available in this repository.
