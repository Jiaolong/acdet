import torch
from torch import nn


class CrossAttention(nn.Module):
    """Cross-Attention block.
    """
    def __init__(self, dim_query=32, dim_key=16, dim_feat=64, groups=1):
        super().__init__()
        self.groups = groups

        # linear transform to get query
        self.p = nn.Conv1d(dim_query, dim_feat, kernel_size=1, stride=1, bias=False)

        # linear transform to get values
        self.t = nn.Conv1d(dim_key, dim_feat, kernel_size=1, stride=1, bias=False)
        # linear transform to get keys
        self.g = nn.Conv1d(dim_key, dim_feat, kernel_size=1, stride=1, bias=False)

        # conv linear
        self.z = nn.Conv1d(dim_feat, dim_query, kernel_size=1, stride=1, groups=self.groups, bias=False)

        # norm (essentially LayerNorm per group)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=dim_query)

        # softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.view(b, c, h).permute(0, 2, 1)  # B x H x C
        proj_key = g  # B x C x H
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        total_energy = energy
        attention = self.softmax(total_energy)  # B x N x N
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h)
        return out

    def forward(self, x, y, masks):
        
        residual = x
        batch_size, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(batch_size, H * W, -1)
        y = y.permute(0, 2, 3, 1).view(batch_size, H * W, -1)

        masks = masks.reshape(batch_size, H * W)
        x = x[masks > 0].t().contiguous().unsqueeze(0)
        y = y[masks > 0].t().contiguous().unsqueeze(0)

        p = self.p(x) # query

        t = self.t(y) # value
        g = self.g(y) # key

        b, c, h = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)

        x = self.z(x)
        x = self.gn(x) # 1 x C  x N
        x = x.squeeze(0).t() # N x C

        # recover shape
        out = x.new_zeros(batch_size, H * W, x.shape[1])
        out[masks > 0] = x
        out = out.permute(0, 2, 1).reshape(batch_size, x.shape[1], H, W).contiguous()
        out = out + residual
        return out 
