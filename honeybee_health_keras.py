# Honey Bee Health Detection using Keras
# Author: Rohit Vincent
# Version: Python 3.6
import os
# Plotting
from math import pi

import numpy as np
import pandas as pd
# Image processing imports
import skimage.io as skimg
import skimage.transform as transform
import tensorflow
from bokeh.io import output_file
from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d, Plot
from bokeh.models.annotations import Title
from bokeh.models.glyphs import ImageURL
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import cumsum
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
# Image data generator
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
# import to split data
from sklearn.model_selection import train_test_split as split

# Settings
np.random.seed(40)
tensorflow.set_random_seed(40)
# Output File for Bokeh Plots
output_file('honey bee.html', title='Health of your Bees')
# Plot for loss
plot_loss = figure(title='Loss during Training & Validation')
# Plot for accuracy
plot_acc = figure(title='Accuracy during Training & Validation')
# Global variables
img_dir = './bee_imgs/'
# Widht of images
width_img = 100
# Height of images
height_img = 100
# Color images haves 3 channels
channels = 3
# Print Log details during training
print_flag = 1
# size of test & validation set 10%
test_size = 0.1
# Batch size
batch_size = 50
# No of epochs
epoch_no = 50

bees = pd.read_csv('../input/bee_data.csv',
                   index_col=False,
                   parse_dates={'datetime': [1, 2]},
                   dtype={'subspecies': 'category', 'health': 'category', 'caste': 'category'})


# Get the size of an image
def get_image_sizes(file):
    # Load the image
    image = skimg.imread(img_dir + file)
    # Return size of the image
    return list(image.shape)


# Read image & Resize it to 100x100 & Return 3 channels of the image
def get_images(file):
    # Load the image
    image = skimg.imread(img_dir + file)
    # Resize all images to a default size
    image = transform.resize(image, (width_img, height_img), mode='reflect')
    # Return only 3 channels even if there are more.
    return image[:, :, :channels]


# Plot line graph for metric (Training + Validation)
def plot_metric(plot, metric):
    plot.line(list(range(1, epoch_no + 1)), hist_metrics[metric], line_color='green', line_width=1,
              legend="Training")
    plot.circle(list(range(1, epoch_no + 1)), hist_metrics[metric], fill_color="green", size=8, legend="Training")
    plot.line(list(range(1, epoch_no + 1)), hist_metrics['val_' + metric], line_color='red', line_width=1,
              line_dash='dotted', legend="Validation")
    plot.circle(list(range(1, epoch_no + 1)), hist_metrics['val_' + metric], fill_color="red", size=8,
                legend="Validation")
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    return plot


# PLot bar count of field
def plot_barcount(health_values, title):
    health = list(health_values.keys())
    counts = list(health_values.values())
    plot = figure(x_range=health, y_range=(0, 3000), title=title, toolbar_location=None, tools="")
    source = ColumnDataSource(data=dict(health=health, counts=counts, color=Spectral6))
    plot.vbar(x='health', top='counts', width=0.9, source=source, legend="health", color='color')
    plot.xaxis.visible = False
    plot.legend.orientation = "vertical"
    plot.legend.location = "top_center"
    return plot


# Scatter plot of image sizes
def plot_imagesize(images):
    # Get pandas file of sizes of each image
    image_sizes = np.stack(images.apply(get_image_sizes))
    # Convert to data frame
    df = pd.DataFrame(image_sizes, columns=['w', 'h', 'c'])
    # randomize radius of each image
    radii = np.random.random(size=df.shape[0]) * 2.5
    # Define figure
    plot = figure(title="Height vs. Width of images", x_range=[0, 300], y_range=[0, 300])
    # Colors based on height & width
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50 + 2 * df['w'], 30 + 2 * df['h'])
    ]
    # Scatter plot
    plot.scatter(df['w'], df['h'], radius=radii,
                 fill_color=colors, fill_alpha=0.6,
                 line_color=None)
    # Axis labels
    plot.xaxis.axis_label = 'Width'
    plot.yaxis.axis_label = 'Height'
    return plot


# Pie chart of bees carrying pollen
def plot_pollen(bee_df):
    # pollen carrying bees
    pollen = bee_df.groupby('pollen_carrying')['pollen_carrying'].count()
    x = {'Not Carrying': pollen[0], 'Carrying Pollen': pollen[1]}
    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'pollen'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = ["#99d594", "#FF0000"]
    plot = figure(title="Pollen Carrying Bees", toolbar_location=None,
                  tools="hover", tooltips="@pollen: @value", x_range=(-0.5, 1.0))
    plot.wedge(x=0, y=1, radius=0.4,
               start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
               line_color="#99d594", fill_color='color', legend='pollen', source=data)
    plot.axis.axis_label = None
    plot.axis.visible = False
    plot.grid.grid_line_color = None
    return plot


#  Preprocess data
def preprocess(data):
    # Cannot impute nans, drop them
    data.dropna(inplace=True)
    # Check images sizes & plot scatter plot of Height vs. Width
    image_plot = plot_imagesize(data['file'])
    # Get image file name from csv & reduce the dataset to bees which have images
    bees_with_images = data['file'].apply(lambda file: os.path.exists(img_dir + file))
    return data[bees_with_images], image_plot


# Balance the dataset for the field
def balance(data, field):
    # Number of samples in each category of health
    category_count = int(len(data) / data[field].cat.categories.size)
    return data.groupby(field, as_index=False).apply(
        lambda index: index.sample(category_count, replace=True)).reset_index(drop=True)


# Check accuracy of testing
def test_accuracy(model, test_X, test_y):
    # Predicted
    predicted = model.predict(test_X)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(test_y.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=test_y.columns))
    test_res = model.evaluate(test_X, test_y.values, verbose=print_flag)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
    # Accuracy by subspecies
    category_acc = np.logical_and((predicted > 0.5), test_y).sum() / test_y.sum()
    health = list(category_acc.keys())
    counts = list(category_acc)
    plot = figure(x_range=health, y_range=(0, 1), title='Accuracy for each health', toolbar_location=None, tools="")
    source = ColumnDataSource(data=dict(health=health, counts=counts, color=Spectral6))
    plot.vbar(x='health', top='counts', width=0.9, source=source, legend="health", color='color')
    plot.xaxis.visible = False
    plot.legend.orientation = "vertical"
    plot.legend.location = "top_center"
    return plot


# Visualize healthy & sick bees
def visualise_bees(bees):
    healthy = bees[bees['health'] == 'healthy'].sample(5)
    xdr = Range1d(start=-100, end=200)
    ydr = Range1d(start=-100, end=200)
    file = img_dir + healthy.iloc[0]['file']
    file1 = img_dir + healthy.iloc[1]['file']
    file2 = img_dir + healthy.iloc[2]['file']
    file3 = img_dir + healthy.iloc[3]['file']
    source = ColumnDataSource(dict(
        url=[file],
        url1=[file1],
        url2=[file2],
        url3=[file3],
        x1=np.linspace(0, 0, 1),
        y1=np.linspace(0, 0, 1),
        x2=np.linspace(100, 0, 1),
        y2=np.linspace(100, 0, 1),
        x3=np.linspace(0, 0, 1),
        y3=np.linspace(100, 0, 1),
        x4=np.linspace(100, 0, 1),
        y4=np.linspace(0, 0, 1),
        w1=np.linspace(100, 100, 1),
        h1=np.linspace(100, 100, 1)

    ))
    title = Title()
    title.text = 'Healthy Bees'
    health_plot = Plot(
        title=title, x_range=xdr, y_range=ydr,
        h_symmetry=False, v_symmetry=False, min_border=0)
    image = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
    image1 = ImageURL(url="url1", x="x2", y="y2", w="w1", h="h1", anchor="center")
    image2 = ImageURL(url="url2", x="x3", y="y3", w="w1", h="h1", anchor="center")
    image3 = ImageURL(url="url3", x="x4", y="y4", w="w1", h="h1", anchor="center")
    health_plot.add_glyph(source, image)
    health_plot.add_glyph(source, image1)
    health_plot.add_glyph(source, image2)
    health_plot.add_glyph(source, image3)
    sick_cat = bees['health'].cat.categories
    sick = []
    for bee in sick_cat:
        if bee == 'healthy': continue
        sick_bee = bees[bees['health'] == bee].sample(1).iloc[0]
        sick.append(img_dir + sick_bee['file'])
    source = ColumnDataSource(dict(
        url=[sick[0]],
        url1=[sick[1]],
        url2=[sick[2]],
        url3=[sick[3]],
        x1=np.linspace(0, 0, 1),
        y1=np.linspace(0, 0, 1),
        x2=np.linspace(100, 0, 1),
        y2=np.linspace(100, 0, 1),
        x3=np.linspace(0, 0, 1),
        y3=np.linspace(100, 0, 1),
        x4=np.linspace(100, 0, 1),
        y4=np.linspace(0, 0, 1),
        w1=np.linspace(100, 100, 1),
        h1=np.linspace(100, 100, 1)

    ))
    title1 = Title()
    title1.text = 'Sick Bees'
    sick_plot = Plot(
        title=title1, x_range=xdr, y_range=ydr,
        h_symmetry=False, v_symmetry=False, min_border=0)
    image = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
    image1 = ImageURL(url="url1", x="x2", y="y2", w="w1", h="h1", anchor="center")
    image2 = ImageURL(url="url2", x="x3", y="y3", w="w1", h="h1", anchor="center")
    image3 = ImageURL(url="url3", x="x4", y="y4", w="w1", h="h1", anchor="center")
    sick_plot.add_glyph(source, image)
    sick_plot.add_glyph(source, image1)
    sick_plot.add_glyph(source, image2)
    sick_plot.add_glyph(source, image3)
    return health_plot, sick_plot


# Visualize sick & healthy bees
health_plot, sick_plot = visualise_bees(bees)
# Plot number of bees containing pollen
pollen_plot = plot_pollen(bees)
# Preprocess CSV file
bees, image_plot = preprocess(bees)
# Split data intro training, validation & test set
train_bees, test_bees = split(bees, test_size=test_size)
train_bees, validate_bees = split(train_bees, test_size=test_size)
# Plot health issues before balancing
plot_before_bal = plot_barcount(dict(train_bees['health'].value_counts()), 'Count of health issues before balancing')
# Balance the dataset for each type of healthy bee
train_bees = balance(train_bees, 'health')
# Plot health issues after balancing
plot_after_bal = plot_barcount(dict(train_bees['health'].value_counts()), 'Count of health issues after balancing')

# Get the images for the bees after preprocessing - LOADS IMAGES
train_X = np.stack(train_bees['file'].apply(get_images))
validate_X = np.stack(validate_bees['file'].apply(get_images))
test_X = np.stack(test_bees['file'].apply(get_images))
# Convert the different health categories to numerical variables
train_y = pd.get_dummies(train_bees['health'])
validate_y = pd.get_dummies(validate_bees['health'])
test_y = pd.get_dummies(test_bees['health'])
# Create Generator with images by randomly rotating,shifting,flipping & zooming
imgDG = ImageDataGenerator(
    rotation_range=180,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
# Modify the training images based on the Generator
imgDG.fit(train_X)
# Stop training if no improvement in accuracy, Check for 20 Epochs
early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=print_flag)

# Save the best model during the training
save_best = ModelCheckpoint('healthy_model'
                            , monitor='val_acc'
                            , verbose=print_flag
                            , save_best_only=True
                            , save_weights_only=True)
# Create model for Convolution Neural Network
model = Sequential()
model.add(Conv2D(16, kernel_size=3, input_shape=(width_img, height_img, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2))
# Add dropouts to the model
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
# Add dropouts to the model
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(train_y.columns.size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# steps per epoch = training size/batch size
steps = np.round(train_X.shape[0] / batch_size, 0)
# Train the model for the training set
history = model.fit_generator(imgDG.flow(train_X, train_y, batch_size=batch_size)
                              , epochs=epoch_no
                              , validation_data=[validate_X, validate_y]
                              , steps_per_epoch=steps
                              , callbacks=[early_stop, save_best])

# Get metrics of training
hist_metrics = history.history
# Plot metrics in chart
# Loss over Training & Validation
plot_loss = plot_metric(plot_loss, 'loss')
plot_acc = plot_metric(plot_acc, 'acc')
# Get the best saved weights
model.load_weights('healthy_model')
# Evaluate the created model & Print accuracy of the model
results = model.evaluate(test_X, test_y.values, verbose=print_flag)
print('Model was created with Loss of ' + str(results[0]) + ' with Accuracy: ' + str(results[1]))
accuracy_plot = test_accuracy(model, test_X, test_y)
# Create gridplot to display all graphs
grid = gridplot([health_plot, sick_plot, plot_before_bal, plot_after_bal, plot_loss, plot_acc, image_plot, pollen_plot,
                 accuracy_plot], ncols=2, plot_width=600, plot_height=600)
# Display grid
show(grid)
