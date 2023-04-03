import sys
sys.path.append('../')
from src import *

# Open the YAML configuration file
SCRIPTDIR = os.path.dirname(__file__)
YAMLFILE = os.path.join(SCRIPTDIR, 'config.yaml')
with open(YAMLFILE, 'r') as file:
    cf = yaml.safe_load(file)

# ---------------------------------------------------------------------------- #
#                        Initial Configuration Variables                       #
# ---------------------------------------------------------------------------- #

augmented = cf['augmented']
epochs = cf['epochs']
target_size = cf['target_size']

# Images Path
main_path_test = cf["main_path_test"]
main_path_train = cf["main_path_train"]
main_path_val = cf["main_path_val"]
folders = cf["folders"]


# batch_size for training
b_size = cf["b_size"]

# Save figures
figs_folder = cf["figs_folder"]

# Save model
checkpoint_path = cf["checkpoint_path"]

# Save history.history
path_save_history = cf["path_save_history"]

# Train parameters
early_stop_patience = cf["early_stop_patience"]
reduce_lr_patience = cf["reduce_lr_patience"]

# ---------------------------------------------------------------------------- #
#                  Create Image Generators and Get Input Shape                 #
# ---------------------------------------------------------------------------- #

image_gen = get_ImageDataGenerator()
image_gen_test = get_ImageDataGenerator()

train_generator = get_generator(image_gen,main_path_train,folders,target_size,b_size)

# Define the input_shape
batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])

print("==> Sparse labels (classes), where 0 = Not-Covid, 1 = COVID-19\n",np.array(batch[1][:5]).astype(int))
input_shape = batch_images[0].shape
print("==> Input Shape: ",input_shape)

# --------- Reset train generator and define validation and test ones -------- #

train_generator = get_generator(image_gen, main_path_train,folders,target_size,b_size,shuffle=True,seed=1)
val_generator = get_generator(image_gen, main_path_val,folders,target_size,b_size,shuffle=True,seed=1)


# ---------------------------------------------------------------------------- #
#                          Build and Compile the Model                         #
# ---------------------------------------------------------------------------- #
model, model_name = get_model(input_shape)
model_name = model_name + '_epochs-%s'%epochs 

print("==> Model %s loaded. \n"%model_name)
print(model.summary())

# -------------------------- Save the model display -------------------------- #
path_save_fig = figs_folder + model_name 
display_model(model, path_save_fig)

# ------------------------ Define path to save model ------------------------ #
path_save_best = checkpoint_path + model_name

# ---------------------------------------------------------------------------- #
#                                  Train Model                                 #
# ---------------------------------------------------------------------------- #
history = train_model(model,train_generator,val_generator, epochs, path_save_best,
                      early_stop_patience,reduce_lr_patience)

# ---------------------------------------------------------------------------- #
#                                 Save history                                 #
# ---------------------------------------------------------------------------- #
path_history = path_save_history + 'history_' + model_name 
save_history(history,path_history)   