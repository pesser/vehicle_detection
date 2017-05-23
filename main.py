import os, urllib.request, sys, math, csv, pickle, glob, time
from zipfile import ZipFile
import numpy as np
import cv2
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.pipeline
import scipy.stats
import skimage.feature
import moviepy.editor

# path where training data will be stored
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok = True)

# path where output images etc. will be stored
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok = True)


def dl_progress(count, block_size, total_size):
    """Progress bar used during download."""
    if total_size == -1:
        if count == 0:
            sys.stdout.write("Unknown size of download.\n")
    else:
        length = 50
        current_size = count * block_size
        done = current_size * length // total_size
        togo = length - done
        prog = "[" + done * "=" + togo * "-" + "]"
        sys.stdout.write(prog)
        if(current_size < total_size):
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")
    sys.stdout.flush()


def download_data():
    """Download data."""
    data = {
            "vehicles.zip": "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip",
            "non_vehicles.zip": "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip"}
    local_data = {}
    for fname, url in data.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print("Downloading {}".format(fname))
            urllib.request.urlretrieve(url, path, reporthook = dl_progress)
        else:
            print("Found {}. Skipping download.".format(fname))
        local_data[fname] = path
    return local_data


def extract_data(path):
    """Extract zip file if not already extracted."""
    with ZipFile(path) as f:
        targets = dict((fname, os.path.join(data_dir, fname)) for fname in f.namelist())
        if not all([os.path.exists(target) for target in list(targets.values())[:10]]): # only check for first 10 for speed
            print("Extracting {}".format(path))
            f.extractall(data_dir)
        else:
            print("Skipping extraction of {}".format(path))
    return targets


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def table_format(row, header = False, width = 20):
    """Format row as markdown table."""
    result = "|" + "|".join(str(entry).center(width) for entry in row) + "|"
    if header:
        result = result + "\n" + "|" + "|".join([width * "=" for _ in row]) + "|"
    return result


def explore_data():
    """Initial exploration of what data looks like. Return filenames for
    both classes and label map."""
    labels = ["vehicles", "non-vehicles"]
    labelmap = {0: "vehicles", 1: "non-vehicles"}
    vehicles_glob = os.path.join(data_dir, "vehicles", "**", "*.png")
    nonvehicles_glob = os.path.join(data_dir, "non-vehicles", "**", "*.png")
    class_fnames = [
            glob.glob(vehicles_glob, recursive = True),
            glob.glob(nonvehicles_glob, recursive = True)]
    n_samples = [len(fnames) for fnames in class_fnames]
    shapes = []
    samples = []
    print(table_format(["label", "size", "shape"], header = True))
    for label, fnames in enumerate(class_fnames):
        indices = np.random.choice(len(fnames), 4*10, replace = False)
        for i in indices:
            fname = fnames[i]
            img = cv2.imread(fname)
            samples.append(img)
            shape = img.shape
        shapes.append(shape)
        print(table_format([labels[label], n_samples[label], shapes[label]]))

    samples = np.stack(samples)
    samples = tile(samples, 2*4, 10)
    cv2.imwrite(os.path.join(out_dir, "datasamples.png"), samples)

    return class_fnames, labelmap


def load_data(class_fnames):
    """Load complete data into memory."""
    X = []
    y = []
    for label, fnames in enumerate(class_fnames):
        for fname in fnames:
            X.append(cv2.imread(fname))
            y.append(label)
    X = np.stack(X)
    y = np.stack(y)
    return X, y


def prepare_data(X, y, n = None):
    """Shuffle and split data."""
    # notice that the training data contains time series data since the GTI
    # data is collected from videos and thus frames from the same video are
    # not independent. It would be better to split based on video but
    # unfortunately this information is not included in the dataset and thus
    # we are leaking data from test set into training set which makes the
    # testing performance a less reliable estimate for the classifiers
    # generalization ability.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size = 0.15)
    if n is not None:
        X_train = X_train[:n,...]
        y_train = y_train[:n,...]
        X_test = X_test[:n,...]
        y_test = y_test[:n,...]
    return (X_train, y_train), (X_test, y_test)


def get_data(n = None):
    """Get data in ready to use format. Parameter determines maximum number
    of samples to return during development."""
    files = download_data()
    for path in files.values():
        extract_data(path)
    class_fnames, labelmap = explore_data()
    X, y = load_data(class_fnames)
    return prepare_data(X, y, n = n)


def features_colorspace(X, colorspace):
    """Convert from BGR to desired colorspace."""
    if colorspace == "BGR":
        return X

    X = np.array(X)

    if colorspace == "HLS":
        cs = cv2.COLOR_BGR2HLS
    elif colorspace == "HSV":
        cs = cv2.COLOR_BGR2HSV
    elif colorspace == "LUV":
        cs =  cv2.COLOR_BGR2LUV
    elif colorspace == "YUV":
        cs = cv2.COLOR_BGR2YUV
    elif colorspace == "YCrCb":
        cs = cv2.COLOR_BGR2YCrCb
    else:
        raise ValueError(colorspace)

    for i in range(X.shape[0]):
        X[i,...] = cv2.cvtColor(X[i,...], cs)

    return X


def features_spatial(X, size, channels):
    """Extract features by resizing and flattening desired channels."""
    features = np.zeros((X.shape[0], size[0]*size[1]*len(channels)))
    for i in range(X.shape[0]):
        img = X[i,...][:,:,channels]
        features[i,...] = cv2.resize(img, size).ravel()
    return features


def features_hist(X, bins, channels):
    """Extract features by calculating histograms over desired channels."""
    features = np.zeros((X.shape[0], bins*len(channels)))
    for i in range(X.shape[0]):
        channel_features = []
        for channel in channels:
            img = X[i,...][:,:,channel]
            hist, bin_edges = np.histogram(img, bins = bins, range = (0, 256))
            channel_features.append(hist)
        features[i,...] = np.concatenate(channel_features)
    return features


def features_hog(X, block_norm, transform_sqrt, channels):
    """Extract features by computing HOG features with desired parameters."""
    orientations = 9
    pixels_per_cell = (8,8)
    cells_per_block = (3,3)

    # calculate size of hog feature vector
    img_shape = X[0,...].shape
    n_blocks = [None, None]
    for i in range(2):
        blocksize = pixels_per_cell[i] * cells_per_block[i]
        n_blocks[i] = int((img_shape[i] - blocksize) / pixels_per_cell[i] + 1)
    n_hog_features = np.prod(n_blocks) * np.prod(cells_per_block) * orientations

    # collect features over images and channels
    features = np.zeros((X.shape[0], n_hog_features*len(channels)))
    for i in range(X.shape[0]):
        channel_features = []
        for channel in channels:
            img = X[i,...][:,:,channel]
            channel_features.append(
                    skimage.feature.hog(
                        img,
                        orientations = orientations,
                        pixels_per_cell = pixels_per_cell,
                        cells_per_block = cells_per_block,
                        block_norm = block_norm,
                        transform_sqrt = transform_sqrt))
        features[i,...] = np.concatenate(channel_features)
    return features


class Transformer(sklearn.base.TransformerMixin):
    """Transformer class for feature extraction. By implementing the sklearn
    Transformer interface we can use it in a pipeline and cross validate its
    parameters jointly with the classifiers parameters."""

    def __init__(self,
            colorspace = "BGR",
            spatial_size = (16,16),
            spatial_channels = [0,1,2],
            hist_bins = 16,
            hist_channels = [0,1,2],
            hog_block_norm = "L1",
            hog_transform_sqrt = False,
            hog_channels = [0,1,2]):
        self.params = dict()
        self.params["colorspace"] = colorspace
        self.params["spatial_size"] = spatial_size
        self.params["spatial_channels"] = spatial_channels
        self.params["hist_bins"] = hist_bins
        self.params["hist_channels"] = hist_channels
        self.params["hog_block_norm"] = hog_block_norm
        self.params["hog_transform_sqrt"] = hog_transform_sqrt
        self.params["hog_channels"] = hog_channels


    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if not k in self.params:
                raise ValueError("{} is not a valid parameter for Transformer.".format(k))
            self.params[k] = v


    def get_params(self, **kwargs):
        return self.params


    def fit(self, X, y = None):
        return self


    def transform(self, X, verbose = False):
        """Transform images to their features."""
        if self.verbose:
            t = Timer()
        X = features_colorspace(X, self.params["colorspace"])
        feature_list = list()
        feature_list.append(
                features_spatial(X, self.params["spatial_size"], self.params["spatial_channels"]))
        feature_list.append(
                features_hist(X, self.params["hist_bins"], self.params["hist_channels"]))
        feature_list.append(
                features_hog(X, self.params["hog_block_norm"], self.params["hog_transform_sqrt"], self.params["hog_channels"]))
        features = np.concatenate(feature_list, axis = 1)
        if self.verbose:
            print("Number of features: {}".format(features.shape[1]))
            print("Time to transform : {:.4e}".format(t.tock()/X.shape[0]))
        return features


class Timer(object):
    """Simple timer class to find performance bottlenecks."""

    def __init__(self):
        self.tick()

    
    def tick(self):
        self.start_time = time.time()


    def tock(self):
        self.end_time = time.time()
        time_since_tick = self.end_time - self.start_time
        self.tick()
        return time_since_tick


def slide_window(
        img,
        x_start_stop=[None, None], y_start_stop=[None, None], 
        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Slide window over image and return all resulting bounding boxes."""
    # If x and/or y start/stop positions not defined, set to image size
    if not x_start_stop[0]:
        x_start_stop[0] = 0
    if not x_start_stop[1]:
        x_start_stop[1] = img.shape[1]
    if not y_start_stop[0]:
        y_start_stop[0] = 0
    if not y_start_stop[1]:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    w = x_start_stop[1] - x_start_stop[0]
    h = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    pps_x = int((1.0 - xy_overlap[0]) * xy_window[0])
    pps_y = int((1.0 - xy_overlap[1]) * xy_window[1])
    # Compute the number of windows in x/y
    n_x = int((w - xy_window[0])/pps_x + 1)
    n_y = int((h - xy_window[1])/pps_y + 1)
    # Initialize a list to append window positions to
    window_list = []
    for i in range(n_y):
        y_pos = i * pps_y + y_start_stop[0]
        for j in range(n_x):
            x_pos = j * pps_x + x_start_stop[0]
            bbox = ((x_pos,y_pos), (x_pos+xy_window[0],y_pos+xy_window[1]))
            window_list.append(bbox)
            
    return window_list


def get_multiscale_windows(img):
    """Return bounding boxes of windows of different scales slid over img
    for likely vehicle positions."""
    window_list = list()
    method = "right"
    if method == "right":
        # for the included video this is fine to improve speed
        # but for other videos, method == "full" should be used
        window_list += slide_window(img,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [620, 620 + 6*96],
                y_start_stop = [385, 385 + 2*96],
                xy_window = (96, 96))
        window_list += slide_window(img,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [620, None],
                y_start_stop = [385, 385 + 2*128],
                xy_window = (128, 128))
    elif method == "full":
        window_list += slide_window(img,
                xy_overlap = (0.75, 0.75),
                x_start_stop = [620 - 6*96, 620 + 6*96],
                y_start_stop = [385, 385 + 2*96],
                xy_window = (96, 96))
        window_list += slide_window(img,
                xy_overlap = (0.75, 0.75),
                y_start_stop = [385, 385 + 2*128],
                xy_window = (128, 128))
    else:
        raise ValueError(method)
    return window_list


def extract_window(img, bbox):
    """Extract patch from window and rescale to size used by classifier."""
    row_begin = bbox[0][1]
    row_end = bbox[1][1]
    col_begin = bbox[0][0]
    col_end = bbox[1][0]
    patch = img[row_begin:row_end, col_begin:col_end, :]
    window = cv2.resize(patch, (64,64))
    return window


def detect(img, window_list, pipeline):
    """Classify all windows within img."""
    #t = Timer()
    windows = []
    for bbox in window_list:
        window = extract_window(img, bbox)
        windows.append(window)
    windows = np.stack(windows)
    detections = pipeline.predict(windows)
    #print("Time to detect: {:.2f}".format(t.tock()))
    return detections


def add_heat(heatmap, bboxes, amount):
    """Add specified amount of heat within each bounding box and avoid
    overflow."""
    for bbox in bboxes:
        window = heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        window[window < 256 - amount] += amount
    return heatmap


def cool_down(heatmap, amount):
    """Cool down complete heatmap by specified amount and avoid
    underflow."""
    heatmap[heatmap <= amount] = 0
    heatmap[heatmap > amount] -= amount
    return heatmap


def midpoint(bbox):
    """Midpoint of bounding box."""
    return (0.5*(bbox[0][0] + bbox[1][0]), 0.5*(bbox[0][1] + bbox[1][1]))


def bbox_dist(bbox1, bbox2):
    """Euclidean distance of bounding boxes' midpoints."""
    d = 0.0
    midpoint1 = midpoint(bbox1)
    midpoint2 = midpoint(bbox2)
    for i in range(2):
        d += (midpoint1[i] - midpoint2[i])**2
    d = np.sqrt(d)
    return d


def update_bbox(prev, new, alpha):
    """Exponential moving average applied to corners of bounding boxes."""
    prev = np.array(prev)
    new = np.array(new)
    avg = alpha * new + (1.0 - alpha) * prev
    avg = np.int_(avg)
    avg_tuple = ((avg[0,0], avg[0,1]), (avg[1,0], avg[1,1]))
    return avg_tuple


class VehicleTracking(object):
    """Vehicle detection pipeline."""

    def __init__(self, clf_pipeline, sample_img):
        self.pipeline = clf_pipeline
        self.sample_img = sample_img
        self.window_list = get_multiscale_windows(self.sample_img)
        self.heatmap = np.zeros(self.sample_img.shape[:2], dtype = np.uint8)

        self.decay = 0.075          # lower values for smoother detections
        self.centroid_radius = 60   # maximum distance in pixels between
                                    # midpoints to consider them equal
        self.averaged_bboxes = []

        self.verbose = 0            # display mode


    def update_bboxes(self, ths):
        """Track bounding boxes given a thresholded heatmap."""
        # find bboxes based on thresholded heatmap
        labels, n_labels = scipy.ndimage.measurements.label(ths)
        detected_bboxes = list()
        for car_label in range(1, n_labels + 1):
            nonzero = (labels == car_label).nonzero()
            bbox = ((np.min(nonzero[1]), np.min(nonzero[0])),
                    (np.max(nonzero[1]), np.max(nonzero[0])))
            detected_bboxes.append(bbox)
        self.detected_bboxes = detected_bboxes

        # match new and previous detections
        N_known = len(self.averaged_bboxes)
        N_new = len(detected_bboxes)
        dmatrix = np.zeros((N_known, N_new))
        for i in range(N_known):
            for j in range(N_new):
                dmatrix[i,j] = bbox_dist(
                        self.averaged_bboxes[i],
                        detected_bboxes[j])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dmatrix)
        # only consider matches whose centroids are close
        mask = dmatrix[row_ind, col_ind] < self.centroid_radius
        matched_row_ind = row_ind[mask]
        matched_col_ind = col_ind[mask]
        # update moving average 
        for i, j in zip(matched_row_ind, matched_col_ind):
            avg_bbox = self.averaged_bboxes[i]
            new_bbox = detected_bboxes[j]
            self.averaged_bboxes[i] = update_bbox(avg_bbox, new_bbox, self.decay)

        # remove bounding boxes which are not present anymore
        self.averaged_bboxes = [bbox for i, bbox in enumerate(self.averaged_bboxes) if i in matched_row_ind]
        # add new bounding boxes
        for j in range(N_new):
            if not j in matched_col_ind:
                self.averaged_bboxes.append(detected_bboxes[j])


    def __call__(self, x):
        """Frame transformation."""
        # convert from moviepy's rgb to opencv's bgr
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        # run detection
        detections = detect(x, self.window_list, self.pipeline)
        detected_windows = [bbox for label, bbox in zip(detections, self.window_list) if label == 0]

        # update heatmap
        amount = 255 // 15
        add_heat(self.heatmap, detected_windows, amount)
        cool_down(self.heatmap, amount)

        # update bounding boxes
        ths_heatmap = np.array(self.heatmap)
        ths_heatmap[ths_heatmap < 6*amount] = 0
        self.update_bboxes(ths_heatmap)

        if self.verbose:
            # show heatmap
            heat_rgb = np.repeat(self.heatmap[:,:,None], 3, axis = 2)
            x = cv2.addWeighted(x, 0.5, heat_rgb, 0.5, 1.0)

            # red for averaged boxes
            for bbox in self.averaged_bboxes:
                cv2.rectangle(x, bbox[0], bbox[1], (0,0,255), 6)

            # blue for current, unsmoothened boxes
            for bbox in self.detected_bboxes:
                cv2.rectangle(x, bbox[0], bbox[1], (255,0,0), 4)

            # green for all positive detections in current frame
            for bbox in detected_windows:
                cv2.rectangle(x, bbox[0], bbox[1], (0,255,0), 2)
        else:
            # picture in picture view of one detection
            if self.averaged_bboxes:
                i = 0
                bbox = self.averaged_bboxes[i]
                window = x[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:]
                margin = 50
                x[margin:margin+window.shape[0],margin:margin+window.shape[1],:] = window
                color = [0,0,0]
                color[i%3] = 255
                cv2.rectangle(x, (margin,margin), (margin+window.shape[1], margin + window.shape[0]), color, 4)
            # all averaged bounding boxes
            for i, bbox in enumerate(self.averaged_bboxes):
                color = [0,0,0]
                color[i%3] = 255
                cv2.rectangle(x, bbox[0], bbox[1], color, 4)

        # convert back to rgb again
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_BGR2RGB)
        return x


if __name__ == "__main__":
    np.random.seed(23) # for reproducibility
    evaluate_model = True

    model_file = os.path.join(out_dir, "model.p")
    if not os.path.isfile(model_file):
        # number of parameters to explore and cross validate - requires
        # three times as many fits
        cv_searches = 10

        # data
        (X_train, y_train), (X_test, y_test) = get_data(n = None)

        timer = Timer()

        # pipeline
        trafo = Transformer()
        scaler = sklearn.preprocessing.StandardScaler()
        clf = sklearn.svm.SVC()
        pipeline = sklearn.pipeline.make_pipeline(trafo, scaler, clf)

        # cross validated randomized search
        channel_choices = [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
        param_grid = dict(
                svc__C = scipy.stats.uniform(loc = 0.1, scale = 1.9),
                svc__kernel = ["linear", "poly", "rbf"],
                svc__degree = [2, 3, 4],
                transformer__colorspace = ["BGR", "HLS", "HSV", "LUV", "YUV", "YCrCb"],
                transformer__spatial_size = [(8,8), (16,16), (32,32)],
                transformer__spatial_channels = channel_choices,
                transformer__hist_bins = [8, 16, 32],
                transformer__hist_channels = channel_choices,
                transformer__hog_block_norm = ["L1", "L2", "L2-Hys"],
                transformer__hog_transform_sqrt = [False, True],
                transformer__hog_channels = channel_choices
                )
        grid_search = sklearn.model_selection.RandomizedSearchCV(
                pipeline,
                param_distributions = param_grid, 
                n_iter = cv_searches,
                n_jobs = 2,
                verbose = 10)

        # training
        timer.tick()
        grid_search.fit(X_train, y_train)
        fit_time = timer.tock()

        # persist best model
        with open(model_file, "wb") as f:
            pickle.dump(grid_search.best_estimator_, f)

        print("Best score: ", grid_search.best_score_)
        print("Best params: ", grid_search.best_params_)
        print("Time to fit    : {:.4f} [s]".format(fit_time))

    # load model
    with open(model_file, "rb") as f:
        pipeline = pickle.load(f)

    if evaluate_model:
        (X_train, y_train), (X_test, y_test) = get_data(n = None)
        pipeline.steps[0][1].verbose = True
        t = Timer()
        test_acc = pipeline.score(X_test, y_test)
        print("Time to predict   : {:.4e}".format(t.tock()/X_test.shape[0]))
        pipeline.steps[0][1].verbose = False
        print("Test accuracy  : {:.4f} [fraction]".format(test_acc))

    # print parameters of model
    params = pipeline.get_params()
    param_keys = sorted([key for key in params.keys()
            if any(key.startswith(component) for component in ["svc__C", "svc__kernel", "transformer__"])])
    param_values = [params[k] for k in param_keys]
    print(table_format(["parameter", "value"], width = 33, header = True))
    for p, v in zip(param_keys, param_values):
        print(table_format([p,v], width = 33))

    # image pipeline
    test_image_fnames = glob.glob("test_images/*.jpg")
    for img_fname in test_image_fnames:
        # load image
        img = cv2.imread(img_fname)
        window_list = get_multiscale_windows(img)
        # visualize windows to be searched for cars
        out_img = np.array(img)
        for bbox in window_list:
            cv2.rectangle(out_img, bbox[0], bbox[1], (200,0,0), 1)
        # run detections on all windows
        detections = detect(img, window_list, pipeline)
        detected_windows = [bbox for label, bbox in zip(detections, window_list) if label == 0]
        # visualize detected windows
        for bbox in detected_windows:
            cv2.rectangle(out_img, bbox[0], bbox[1], (0,255,0), 6)
        cv2.imwrite(os.path.join(out_dir, "detected_" + os.path.basename(img_fname)), out_img)
        # create a heatmap based on detections
        heatmap = np.zeros(img.shape[:2])
        add_heat(heatmap, detected_windows, 1)
        heatmap_vis = np.uint8(255 * heatmap / np.max(heatmap))
        cv2.imwrite(os.path.join(out_dir, "heat_" + os.path.basename(img_fname)), heatmap_vis)
        # thresholded heatmap for increased robustness
        heatmap_ths = np.uint8(255 * (heatmap > 1))
        cv2.imwrite(os.path.join(out_dir, "ths_" + os.path.basename(img_fname)), heatmap_ths)
        # extract bounding boxes
        labels, n_labels = scipy.ndimage.measurements.label(heatmap_ths)
        print("{}: {} cars found.".format(img_fname, n_labels))
        for car_label in range(1, n_labels + 1):
            nonzero = (labels == car_label).nonzero()
            bbox = ((np.min(nonzero[1]), np.min(nonzero[0])),
                    (np.max(nonzero[1]), np.max(nonzero[0])))
            cv2.rectangle(out_img, bbox[0], bbox[1], (0,0,255), 8)
        cv2.imwrite(os.path.join(out_dir, "bounded_" + os.path.basename(img_fname)), out_img)
    print("Number of windows: {}".format(len(window_list)))

    # video pipeline
    for verbose in [True, False]:
        for clip_fname in ["project_video.mp4"]:
            out_clip_fname = "out_"
            if verbose:
                out_clip_fname += "verbose_"
            out_clip_fname += clip_fname
            clip = moviepy.editor.VideoFileClip(clip_fname)
            sample = clip.get_frame(0)
            tracker = VehicleTracking(pipeline, sample)
            tracker.verbose = verbose
            out_clip = clip.fl_image(tracker)
            out_clip.write_videofile(os.path.join(out_dir, out_clip_fname), audio = False)
