import os
import torch

class CFG:
    # ====== General ======
    project_name = "ResNet18_CBAM_MGA"
    seed = 42
    seed_list = [42, 123, 2025, 777, 999]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Data ======
    data_root = "/data1/lidc-idri/slices"
    bbox_csv = "/home/iujeong/lung_cancer/csv/allbb_noPoly.csv"
    input_size = (224, 224)

    # ====== Dataloader ======
    num_workers = 4
    batch_size = 16
    pin_memory = True

    # ====== Model ======
    model_name = "ResNet18_CBAM"
    num_classes = 2

    # ====== Train ======
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    scheduler = "cosine"  # or 'none', 'cosine', 'step'
    class_weights = [0.6653, 0.3347]

    # ====== TTA ======
    use_tta = True

    # ====== MGA (Mask-Guided Attention) ======
    use_mga = False
    lambda_mga_initial = 0.1
    lambda_mga_final = 0.5
    use_rotate_mask = False    #True  # False 하면 회전 안 시킴

    # ====== Save & Logging ======
    save_dir = os.path.join(os.getcwd(), "logs")
    # model_save_name = "r18_cbam_mga_sc_weight.pth"
    log_interval = 10

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.save_dir, exist_ok=True)

    @classmethod
    def as_dict(cls):
        return {k: getattr(cls, k) for k in dir(cls) if not k.startswith("__") and not callable(getattr(cls, k))}
