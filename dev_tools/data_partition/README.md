# Data Partitioning

By 'partitioning', we mean 'selecting which files end up in the train split and which in the test split'. This folder will contain a collection of scripts to take arbitrarily arranged dataset files and storing a partition in a standard format

# Standard format

Each script in this folder should be callable like:

    python3 some_partition.py /path/to/data/folder partition_policy extra_options

where `partition_policy` is something like `{"frame_percent": 0.75}`, and should create a folder in `/path/to/data/folder` called `.partition` that has the following structure:

    /path/to/data/folder/
        .partition/
            <partition_option1>/
                train.csv
                test.csv
            <partition_option2>/
                train.csv
                test.csv
            ...

The partition option can be anything, and the name of the folder can be anything. The two csv files in each folder should be formatted as follows:

    "video_filename1","label_filename1"
    "video_filename2","label_filename2"
    "video_filename3","label_filename3"
    ...

*note*: both the video filenames and label filenames are relative to the `/path/to/data/folder/` to make the partition itself portable with the data folder.

## partition_policy

`partition_policy` represents the rule for splitting training/validation. The value given is a proportion of the whole and represent a minimum going into the training set. The below examples mean that at LEAST 80% of the split goes to the training set. This happens in cases where the videos don't divide evenly e.g. for 11 videos, 9 go to the training set.
This is most cases.
Possible partition policies:
    {"frame_percent": 0.8}
       Split based on number of frames - 80% for training_set
    {"video_percent": 0.8}
       Split based on video count - 80% for training_set
