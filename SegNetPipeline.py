# Load all functions used for SegNet
exec(open("SegNetHelpers.py").read())

################################################################################
## I M P O R T S



################################################################################
## H E L P E R S

def LoadSegNetModel(model_filename, verbose = False):
    if verbose:
        CheckTensorflowGPU()
        # check that model Keras version is same as local Keras version
        CheckKerasVersion(model_filename)
    model = None
    model = load_model(model_filename)
    if verbose:
        model.summary()
    return model

def GetAllClassesFromSegNet(image, model):
    image_in = Normalize(image)
    image_in = PreprocessInput(image_in)
    return model.predict(image_in[None, :, :, :], batch_size = 1)

def GetClassFromSegNet(image, model, num_class = 10):
    # Default: 10 = LaneMarkings
    output_image = GetAllClassesFromSegNet(image, model)
    class_image  = output_image[:,:,num_class].reshape(image.shape[0], image.shape[1])
    return class_image
