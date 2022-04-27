import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ELM_myself import HiddenLayer

# from DELM_myself import HiddenLayer
data = pd.read_csv('r_test.csv')
x = data.x.values.reshape(-1, 1)
y = data.y.values.reshape(-1, 1)

my_EML = HiddenLayer(x, 5)
my_EML.regressor_train(y)
x_test = np.linspace(0.9, 5.02, 100).reshape(-1, 1)
y_test = my_EML.regressor_test(x_test)
plt.plot(x_test, y_test)
plt.scatter(x, y)
plt.title('EML_regress')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# my_DEML = HiddenLayer(x,10)
# my_DEML.regressor_train(y)
# x_test = np.linspace(0.9,5.02,100).reshape(-1,1)
# y_test = my_DEML.regressor_test(x_test)
# plt.plot(x_test,y_test)
# plt.scatter(x,y)
# plt.title('DEML_regress')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
