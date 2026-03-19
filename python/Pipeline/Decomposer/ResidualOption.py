from enum import Enum


class ResidualOption(Enum):
    IGNORE = "ignore"
    OVERWRITE_ORIGINAL_FEATURE = "overwrite_original_feature"
    NEW_FEATURE = "new_feature"