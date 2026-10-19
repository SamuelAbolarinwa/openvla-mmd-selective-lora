"""Load pinned local OpenVLA code and processing assets consistently."""
def load_processor(path):
    from check_processor import load_processor as load
    return load(path)

def load_base(path, **kwargs):
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    config = OpenVLAConfig.from_pretrained(path, local_files_only=True)
    return OpenVLAForActionPrediction.from_pretrained(
        path, config=config, local_files_only=True, **kwargs)
