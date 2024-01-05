import os.path

LOG_NAME = ".log.txt"

def _logpath(dirp : str) -> str:
    return os.path.join(dirp, LOG_NAME)

def append_to_log(flight_dir : str, img_id : str) -> None:
    fpath = _logpath(flight_dir)
    with open(fpath, "a") as f:
        f.write(img_id)
    return None

def log_exists(flight_dir : str) -> bool:
    return os.path.exists(_logpath(flight_dir))

def read_log(flight_dir : str) -> list:
    f = _logpath(flight_dir)
    with open(f, 'r') as file:
        content = [line.rstrip() for line in file]
    return content

def get_missing_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    not_touched = img_set - log_set
    return list(not_touched)

def get_processed_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    procesed = img_set.intersection(log_set)
    return list(procesed)

def get_unknown_files(image_list : list, log_list : list) -> list:
    img_set = set(image_list)
    log_set = set(log_list)
    unknown = log_set - img_set
    return unknown

def fnames_to_fids(path_list : list) -> list:
    return [os.path.basename(image_path).split(".")[0] for image_path in path_list]