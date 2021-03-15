import os
from scr.mypipe.utils import Util


# decorator
def cach_feature(feature_name, directory):
    def _feature(func):
        def wrapper(*args, **kwargs):
            file_path = os.path.join(directory, feature_name + ".pkl")
            if os.path.isfile(file_path):
                output_df = Util.load(file_path)

            else:
                output_df = func(*args, **kwargs)
                # output_df = reduce_mem_usage(output_df, False) # exp027以降CO
                Util.dump(output_df, file_path)
            return output_df

        return wrapper

    return _feature
