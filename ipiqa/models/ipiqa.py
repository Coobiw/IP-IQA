import clip
import torch.nn as nn
import torch
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.utils import TextAttentionPool2d, interpolate_pos_embed, disabled_train, freeze_module, MLPHead

from ipiqa.common.registry import registry

@registry.register_model("ipiqa")
class IPIQA(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/ipiqa.yaml",
    }
    def __init__(
                self,
                base_ckpt='', # your path for clip resnet (default in `ipiqa.yaml`: cache/ckpt/clip/openai/RN50.pt)
                input_resolution=512,
                output_dim=None,
                use_mlp_head=False,
                dropout_rate=0.,
                freeze_text=True,
                head_scale=None,
                qa_token=False,
        ):
        super().__init__()
        self.resnet50 = clip.load(base_ckpt, device="cpu")[0].visual

        self.txt_model = clip.load(base_ckpt, device="cpu")[0].transformer
        self.wte = clip.load(base_ckpt, device="cpu")[0].token_embedding
        self.ln_final = clip.load(base_ckpt, device="cpu")[0].ln_final
        self.txt_pos = clip.load(base_ckpt, device="cpu")[0].positional_embedding
        self.text_projection = clip.load(base_ckpt, device="cpu")[0].text_projection

        self.dtype = self.resnet50.conv1.weight.dtype

        self.feature_dim = self.resnet50.attnpool.c_proj.out_features
        self.resnet50.attnpool.positional_embedding = nn.Parameter(
                interpolate_pos_embed(self.resnet50.attnpool.positional_embedding,input_resolution=input_resolution))
        self.attnpool = self.resnet50.attnpool
        self.resnet50.attnpool = nn.Identity()
        self.txt_attnpool = TextAttentionPool2d(input_resolution//32,embed_dim=2048,txt_dim=1024,num_heads=32,output_dim=1024)

        if use_mlp_head and output_dim:
            self.head = MLPHead(self.feature_dim*2,output_dim,dropout_rate)
        else:
            self.head = nn.Linear(self.feature_dim*2,output_dim) if output_dim else nn.Identity()

        if freeze_text:
            freeze_module(self.txt_model)
            if not qa_token:
                freeze_module(self.wte)
            else:
                print('use qa-token, unfreeze wte ...')
            freeze_module(self.ln_final)
            freeze_module(self.txt_pos)
            freeze_module(self.text_projection)

        self.head_scale = head_scale

    def forward(self,x,text):
        # import pdb;pdb.set_trace()
        txt_feat = self.encode_text(text)
        feat = self.resnet50(x)
        global_visual = self.attnpool(feat)
        global_txt = self.txt_attnpool(feat,txt_feat)
        global_feat = torch.cat([global_visual,global_txt],dim=-1)
        return self.head(global_feat)

    def encode_text(self,text):
        text = clip.tokenize(text,context_length=77,truncate=True).cuda()
        x = self.wte(text).type(self.dtype)
        x = x + self.txt_pos.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_map = self.txt_model(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        if self.head_scale:
            p_head = []
            p_head_non_wd = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if self.head_scale and 'head' in n:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_head_non_wd.append(p)
                else:
                    p_head.append(p)
            else:
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
        if self.head_scale:
            optim_params = [
                {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
                {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
                {"params": p_head, "weight_decay": weight_decay, "lr_scale": self.head_scale},
                {"params": p_head_non_wd, "weight_decay": 0, "lr_scale": self.head_scale},
            ]
            print(f"head scale: {self.head_scale}")
        else:
            optim_params = [
                {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
                {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
            ]
        return optim_params

    @classmethod
    def from_config(cls, cfg):
        base_ckpt = cfg.get('base_ckpt','cache/ckpt/clip/openai/resnet/RN50.pt')
        input_resolution = cfg.get("input_resolution",512)
        output_dim = cfg.get("output_dim",None)
        freeze_text = cfg.get("freeze_text",True)
        unfreeze_wte = cfg.get("unfreeze_wte",False)
        head_scale = cfg.get('head_scale',None)
        use_mlp_head = cfg.get('use_mlp_head',False)
        dropout_rate = cfg.get('dropout_rate',0.)

        model = cls(base_ckpt,input_resolution,output_dim,use_mlp_head,dropout_rate,freeze_text,head_scale,unfreeze_wte)

        load_finetuned = cfg.get("load_finetuned",False)  # you've loaded the clip weight in `__init__` func
        if load_finetuned:
            model.load_checkpoint_from_config(cfg)

        return model