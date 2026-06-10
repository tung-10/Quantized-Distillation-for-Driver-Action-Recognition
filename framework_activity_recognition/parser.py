import os
import pandas as pd
import sys
from framework_activity_recognition.processing import extractFilesFromDirWhichMatchList


# ─────────────────────────────────────────────
# Drive&Act parsers (unchanged)
# ─────────────────────────────────────────────

def parseDriveNActSplitFiles(split_nr=0, folder_splits="/cvhci/data/activity/Pakos/vpfaefflin/activities_3s/",
    data_path="/cvhci/data/activity/Pakos/final_dataset/pakos_videos_ids_1/",
        views=["a_column_co_driver"], new_suffix=".mp4"):
    """
    This method reads the path to Drive&Act train and validation data with given split, path and views and convert the paths into pandas DataFrame instance
    Arguments:
        split_nr: split of the data
        folder_splits: path to folder containing splits
        data_path: path to video data
        views: list of type of view
        new_suffix: new suffix for the videos
    """
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for view in views:
        train_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".train.csv"
        test_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".val.csv"

        train_file = pd.read_csv(train_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])
        test_file = pd.read_csv(test_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])

        train_df = pd.concat([train_df, replacePathInDf(train_file, data_path, new_suffix)], ignore_index=True, copy=False)
        test_df = pd.concat([test_df, replacePathInDf(test_file, data_path, new_suffix)], ignore_index=True, copy=False)

    print("get ", str(len(train_df.index)), " training data from ", str(len(train_file['file_id'].unique())), " video files")
    print("get ", str(len(test_df.index)), " validation data from ", str(len(test_file['file_id'].unique())), " video files")

    return train_df, test_df


def parseDriveNActTestSplitFiles(split_nr=0, folder_splits="/cvhci/data/activity/Pakos/vpfaefflin/activities_3s/",
    data_path="/cvhci/data/activity/Pakos/final_dataset/pakos_videos_ids_1/",
        views=["a_column_co_driver"], new_suffix=".mp4"):
    """
    This method reads the path to Drive&Act test data with given split, path and views and convert the paths into pandas DataFrame instance
    Arguments:
        split_nr: split of the data
        folder_splits: path to folder containing splits
        data_path: path to video data
        views: list of type of view
        new_suffix: new suffix for the videos
    """
    test_df = pd.DataFrame()

    for view in views:
        test_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".test.csv"

        test_file = pd.read_csv(test_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])

        test_df = pd.concat([test_df, replacePathInDf(test_file, data_path, new_suffix)], ignore_index=True, copy=False)

    print("get ", str(len(test_df.index)), " test data from ", str(len(test_file['file_id'].unique())), " video files")

    return test_df


def replacePathInDf(df, data_path, new_suffix):
    """
    This method replaces path in the DataFrame df with path to the data and add new suffix in the end of the path
    Arguments:
        df: DataFrame instance with paths to be replaced
        data_path: path to the video data
        new_suffix: new suffix of the video data
    """
    df_output = pd.DataFrame()
    unique_files = df['file_id'].unique()
    for video_path in unique_files:
        new_file_path = data_path + video_path + new_suffix
        if os.path.isfile(new_file_path):
            file_rows = df[df['file_id'] == video_path].copy()
            file_rows['file_id'] = new_file_path
            df_output = pd.concat([df_output, file_rows], ignore_index=True, copy=False)
        else:
            error_message = "File " + new_file_path + " does not exist!"
            sys.exit(error_message)

    return df_output


# ─────────────────────────────────────────────
# Custom dataset parser (new)
# ─────────────────────────────────────────────

def parseCustomSplitFiles(data_path: str, train_file: str, test_file: str):
    """
    Parse train and test CSV/TXT split files for a custom dataset.

    Expected CSV format (with or without header):
        clip_path,label
        /data/videos/clip001.mp4,walking
        /data/videos/clip002.mp4,sitting

    Arguments:
        data_path:   folder containing the split files
        train_file:  filename of the training split (e.g. "train.csv")
        test_file:   filename of the test/val split  (e.g. "val.csv")

    Returns:
        train_df, test_df — DataFrames with columns ['clip_path', 'label']
    """
    train_df = _read_split_csv(os.path.join(data_path, train_file))
    test_df  = _read_split_csv(os.path.join(data_path, test_file))

    print(f"Train: {len(train_df)} clips  |  {train_df['label'].nunique()} classes")
    print(f"Val/Test: {len(test_df)} clips  |  {test_df['label'].nunique()} classes")

    return train_df, test_df


def parseCustomTestSplitFiles(data_path: str, test_file: str):
    """
    Parse a test-only split file (no train set needed, e.g. for inference).

    Arguments:
        data_path:  folder containing the split file
        test_file:  filename of the test split (e.g. "test.csv")

    Returns:
        test_df — DataFrame with columns ['clip_path', 'label']
                  label column will be 'unknown' if not present in file.
    """
    test_df = _read_split_csv(os.path.join(data_path, test_file))

    if 'label' not in test_df.columns:
        test_df['label'] = 'unknown'

    print(f"Test: {len(test_df)} clips")

    return test_df


def _read_split_csv(csv_path: str) -> pd.DataFrame:
    """
    Internal helper — read a split CSV robustly.
    Handles: with/without header, 1 or 2 columns, comma or whitespace separator.
    Validates that every clip_path actually exists on disk.
    """
    if not os.path.isfile(csv_path):
        sys.exit(f"Split file not found: {csv_path}")

    # Auto-detect separator from first line
    first_line = open(csv_path).readline().strip()
    sep = "," if "," in first_line else r"\s+"

    df = pd.read_csv(csv_path, header=None, sep=sep, engine="python")

    # Drop header row if the file has a literal header
    if str(df.iloc[0, 0]).strip() in ("clip_path", "file_id", "path"):
        df = df.iloc[1:].reset_index(drop=True)

    # Handle 1-column (no label) or 2-column (clip + label) files
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["clip_path", "label"]
    else:
        df.columns = ["clip_path"]
        df["label"] = "unknown"

    df["clip_path"] = df["clip_path"].astype(str).str.strip()
    df["label"]     = df["label"].astype(str).str.strip()

    # Validate paths
    missing = df[~df["clip_path"].apply(os.path.exists)]["clip_path"].tolist()
    if missing:
        msg = f"{len(missing)} clip(s) not found on disk, e.g.:\n  " + "\n  ".join(missing[:5])
        sys.exit(msg)

    return df