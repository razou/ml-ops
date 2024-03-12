from pprint import pprint
import requests

col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
             'DiabetesPedigreeFunction', 'Age']
values = [
    [10, 101, 76, 48, 180, 32.9, 0.171, 63],
    [2, 122, 70, 27, 0, 36.8, 0.34, 27],
    [5, 121, 72, 23, 112, 26.2, 0.245, 30],
    [1, 126, 60, 0, 0, 30.1, 0.349, 47],
    [1, 93, 70, 31, 0, 30.4, 0.315, 23]
]

test_data = []
for v in values:
    test_data.append(dict(zip(col_names, v)))

endpoint = "http://127.0.0.1:8000/predict"
threshold_value = 0.5

if __name__ == "__main__":
    for item in test_data:
        try:
            response = requests.post(url=endpoint, json=item)
            if response.status_code == 200:
                print("Patient information: ")
                print(item)
                print("Diagnostic (probability of having cancer): ")
                pprint(response.json())
                print("-" * 20)
        except Exception as e:
            print(e)
