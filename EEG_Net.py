import os
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

# --------------------------------------------------------------------------------
# EEGNet MODEL 
# --------------------------------------------------------------------------------
def EEGNet(
    nb_classes=1,  # clasificare binara
    Chans=8,       # numar canale EEG
    Samples=3750,  # samples/trial
    dropoutRate=0.5,
    kernLength=64,
    F1=8,          # nr filtre temporale
    D=2,           # depth multiplier
    F2=16          # nr filtre spatiale
):
    """
    Implementare (https://arxiv.org/abs/1611.08024)
    adapted for channels_last (NHWC) on CPU devices.
    
    Input shape: (batch, Samples, Chans, 1)
    Example: (batch, 3750, 8, 1)
    """
    
    input1 = layers.Input(shape=(Samples, Chans, 1))

    # Block 1: Conv2D + DepthwiseConv2D
    block1 = layers.Conv2D(
        filters=F1,
        kernel_size=(kernLength, 1),
        padding='same',
        use_bias=False
    )(input1)
    block1 = layers.BatchNormalization()(block1)

    block1 = layers.DepthwiseConv2D(
        kernel_size=(1, Chans),
        depth_multiplier=D,
        use_bias=False,
        depthwise_constraint=tf.keras.constraints.max_norm(1.)
    )(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((4, 1))(block1)
    block1 = layers.Dropout(dropoutRate)(block1)

    # Block 2: SeparableConv2D
    block2 = layers.SeparableConv2D(
        filters=F2,
        kernel_size=(16, 1),
        padding='same',
        use_bias=False
    )(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((8, 1))(block2)
    block2 = layers.Dropout(dropoutRate)(block2)

    # Classification
    flatten = layers.Flatten()(block2)
    dense = layers.Dense(nb_classes, activation='sigmoid')(flatten)

    model = tf.keras.Model(inputs=input1, outputs=dense)
    return model

# --------------------------------------------------------------------------------
# scurtate primele 2.5 sec, 625 samples shape (8,N)
# --------------------------------------------------------------------------------
def load_raw_trials(path, label, skip_samples=625, chunk_size=3875):

    data_list, labels = [], []
    for fname in os.listdir(path):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(path, fname)).astype(np.float32)
            # 8 canale si min 3875 samples
            if arr.shape[0] == 8 and arr.shape[1] >= (skip_samples + chunk_size):
                arr = arr[:, skip_samples : skip_samples + chunk_size]
                data_list.append(arr)
                labels.append(label)
    return data_list, labels

# --------------------------------------------------------------------------------
#  Datele pt EEGNet: shape => (N, 3750, 8, 1)
# --------------------------------------------------------------------------------
def prepare_eegnet_format(data):
    """
    (8, 3750).
    T (3750, 8)  timp pe axis=0, channels=1,
    extra axis => (3750, 8, 1) pt 'channels_last' input.
    stack=> (N, 3750, 8, 1).
    """
    trials_formatted = []
    for arr in data:
        # arr shape: (8, 3750)
        arr_T = arr.T  # shape => (3750, 8)
        arr_T = arr_T[:, :, np.newaxis]  # => (3750,8,1)
        trials_formatted.append(arr_T)
    return np.array(trials_formatted, dtype=np.float32)

# --------------------------------------------------------------------------------
# MAIN: 5-FOLD STRATIFIED CV
# --------------------------------------------------------------------------------
if __name__ == "__main__":
   
    liked_path = "eeg_recordings/user_3/toate/liked/"
    disliked_path = "eeg_recordings/user_3/toate/disliked/"

    # 1) Load data
    liked_data, liked_labels = load_raw_trials(liked_path, label=1)
    disliked_data, disliked_labels = load_raw_trials(disliked_path, label=0)

    # 2) Concatenare
    X_raw = liked_data + disliked_data
    y = np.array(liked_labels + disliked_labels)

    # 3) EEGNet=> shape (N, 3750, 8, 1)
    X_eegnet = prepare_eegnet_format(X_raw)
    print("EEGNet input shape:", X_eegnet.shape)

    # 4) Stratified K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    train_accs = []
    test_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_eegnet, y), start=1):
        print(f"\n=== Fold {fold} ===")
        # split
        X_train, y_train = X_eegnet[train_idx], y[train_idx]
        X_test,  y_test  = X_eegnet[test_idx],  y[test_idx]

        # 5) fresh EEGNet
        model = EEGNet(Samples=3875, Chans=8)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 6) Train
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50, batch_size=16, verbose=0,
        )

        # Evaluare
        tr_loss, tr_acc = model.evaluate(X_train, y_train, verbose=0)
        ts_loss, ts_acc = model.evaluate(X_test,  y_test,  verbose=0)

        train_accs.append(tr_acc)
        test_accs.append(ts_acc)

        print(f"Train Acc: {tr_acc:.4f} | Test Acc: {ts_acc:.4f}")

    # 7) Results
    print("\n=== 5-Fold CV Results ===")
    print(f"Train Accuracies: {train_accs}")
    print(f"Test Accuracies:  {test_accs}")
    print(f"Mean Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Mean Test:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
