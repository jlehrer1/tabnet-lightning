import torch 
import numpy as np 
import os, sys 
import tabnet 
import shutil 
import json 

import torch.nn.functional as F 

import pytorch_lightning as pl 
from scipy.sparse import csc_matrix 
from pathlib import Path 
from pytorch_tabnet.utils import (
    PredictDataset,
    create_explain_matrix,
    validate_eval_set,
    create_dataloaders,
    define_device,
    ComplexEncoder,
    check_input,
    check_warm_start
)

from pytorch_tabnet.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback,
)

class TabNetLightning(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.network = tabnet.tab_network.TabNet(*args, **kwargs)
        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def forward(self, x):
        return self.base_model.forward(x)

    def training_step(self, batch, batch_idx):
        pass 

    def validation_step(self, batch, batch_idx):
        pass 

    def test_step(self, batch, batch_idx):
        pass 

    def configure_optimizers(self):
        pass 

    def explain(self, loader, normalize=False):
        self.network.eval()
        res_explain = []

        for batch_nb, data in enumerate(loader):
            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(),
                                                   self.reducing_matrix)
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, res_masks

    def _compute_feature_importances(self, dataloader):
        """Compute global feature importance.
        Parameters
        ----------
        loader : `torch.utils.data.Dataloader`
            Pytorch dataloader.
        """
        M_explain, _ = self.explain(dataloader, normalize=False)
        sum_explain = M_explain.sum(axis=0)
        feature_importances_ = sum_explain / np.sum(sum_explain)
        return feature_importances_

    def save_model(self, path):
        """Saving TabNet model in two distinct files.
        Parameters
        ----------
        path : str
            Path of the model.
        Returns
        -------
        str
            input filepath with ".zip" appended
        """
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"