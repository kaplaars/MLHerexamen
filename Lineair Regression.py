import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)

#importing data
raw_dataset = pd.read_csv('full.csv', usecols= ['Survived','Pclass','Sex','Age','SibSp','Parch'])
dataset = raw_dataset.copy()
dataset.tail()
print(raw_dataset)

#cleaning data
print(dataset.isna().sum())
dataset = dataset.dropna()

dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})

#split dataset
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#inspect data
sns.pairplot(train_dataset[['Survived','Pclass','Sex','Age','SibSp','Parch']], diag_kind='kde')
plt.show()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Survived')
test_labels = test_features.pop('Survived')

#normalization
print(train_dataset.describe().transpose()[['mean', 'std']])
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

#regression
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model.predict(train_features[:6])
linear_model.layers[1].kernel
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='mean_absolute_error')
history = linear_model.fit(
    train_features, train_labels,
    epochs=1000,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [Survival]')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

test_results = {}
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=2)
print(pd.DataFrame(test_results, index=['Mean absolute error [Survival]']).T)
test_predictions = linear_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Survival]')
plt.ylabel('Predictions [Survival]')
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()