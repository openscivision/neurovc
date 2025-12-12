from neurovc.contrib.flowmag.flow_models.ARFlow.models.pwclite import PWCLite


def get_model(cfg):
    if cfg.type == "pwclite":
        model = PWCLite(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
