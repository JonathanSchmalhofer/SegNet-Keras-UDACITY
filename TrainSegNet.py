# Load all functions used for SegNet
exec(open("SegNetHelpers.py").read())

CheckTensorflowGPU()

### Configuration
dataset_repo_path            = '/input/CamVid'
output_path                  = '/output'
log_dir                      = output_path + '/logs'
model_filename               = output_path + '/model.h5'                           # output filename for the trained model
model_checkpoint_filename    = output_path + '/model_checkpoint.hdf5'              # output filename for the best checkpoint during training

# Training
train_data_path_list    = []
train_data_path_list.append('/input/CamVid/data.txt') # CamVid

###########################
## Tunable Hyperparameters
num_epochs = 30
batch_size = 2
percentage_validation = 0.33
load_and_train_existing = False
load_checkpoint_not_final_model = False

###########################
## Load & Prepare Dataset

# Load all CSV-files
train_data   = []
train_labels = []
valid_data   = []
valid_labels = []
for data_path in train_data_path_list:
    train_data, train_labels = AppendDataAndLabels(train_data, train_labels, dataset_repo_path, data_path)

# Split into Training & Validation
train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size = percentage_validation)

QualityCheckLabels(train_labels)
QualityCheckLabels(valid_labels)


# Let's use Generators
train_generator = Generator(train_data, train_labels, batch_size = batch_size)
valid_generator = Generator(valid_data, valid_labels, batch_size = batch_size)

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]


# saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath=model_checkpoint_filename, verbose=1, save_best_only=True)

#
tensorboard_logger = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)


###########################
## Build & Train Model

t_train_start = time.time()

model = None
if load_and_train_existing:
    if load_checkpoint_not_final_model:
        file_to_load = model_checkpoint_filename
    else:
        file_to_load = model_filename
    model = load_model(file_to_load)
    model.summary()
else:
    model = GetSegNetArchitecture(input_shape=(720, 960, 3), batch_size = batch_size)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

#BackupExistingModelFile(model_checkpoint_filename) # Backup the Checkpoint, as new checkpoints might be created during training
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch = 2*len(train_data),
                                     validation_data = valid_generator,
                                     nb_val_samples = 2*len(valid_data),
                                     nb_epoch = num_epochs,
                                     class_weight = class_weighting,
                                     callbacks = [checkpointer,tensorboard_logger],
                                     verbose = 1)

elapsed_time = time.time() - t_train_start
print("Elapsed time for training {} min...".format(elapsed_time/60))


###########################
## Post-Processing Stuff

#BackupExistingModelFile(model_filename)
model.save(model_filename)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.savefig('training_results.png', bbox_inches="tight")