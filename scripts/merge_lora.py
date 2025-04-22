from vla.cogactvla_eagle_dual_sys_v2_meta_query_v2 import load_vla
import torch

def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

if __name__ == "__main__":
    vla = load_vla(
                "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/head_balation/sys12_meta_query_action_only_sync_pretraining_v2_query_64_mlp_lora--image_aug/checkpoints/step-036000-epoch-09-loss=0.1682.pt", 
                hf_token="REMOVED_TOKEN", 
                load_for_training=True, 
                stage = "stage1",
                )
    merged_model = vla.vlm.language_model.merge_and_unload()
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
                    "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/head_balation/sys12_meta_query_action_only_sync_pretraining_v2_query_64_mlp_lora--image_aug/checkpoints/step-036000-epoch-09")