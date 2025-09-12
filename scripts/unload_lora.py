from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla
import torch

def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

if __name__ == "__main__":
    vla = load_vla(
                "path/to/checkpoint.pt", 
                load_for_training=True, 
                stage = "stage1",
                )

    vla.vlm.language_model.save_pretrained("path/to/save/lora/modual")
    merged_model = vla.vlm.language_model.unload()
    vla.vlm.language_model = merged_model

    model_state_dicts = {
        mkey: get_nested_attr(vla, mkey).state_dict()
        for mkey in vla.all_module_keys
    }

    for key in list(model_state_dicts.keys()):
        if key.startswith("vlm."):
            value = model_state_dicts.pop(key)
            model_state_dicts[key[4:]] = value
    
    torch.save(dict(model = model_state_dicts), 
                    "path/to/save/the/unloaded.pt") # The model only contains pretrained action head and needs further finetuning for deploy.