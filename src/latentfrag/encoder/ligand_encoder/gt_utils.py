import torch


class PlaceHolder:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y if y is not None else torch.zeros(size=(self.X.shape[0], 0), dtype=torch.float, device=X.device)

    def type_as(self, x: torch.Tensor):
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

    def clone(self):
        if self.y is not None:
            return PlaceHolder(self.X.clone(), self.E.clone(), self.y.clone())
        else:
            return PlaceHolder(self.X.clone(), self.E.clone())

    def detach(self):
        if self.y is not None:
            return PlaceHolder(self.X.clone().detach(), self.E.clone().detach(), self.y.clone().detach())
        else:
            return PlaceHolder(self.X.clone().detach(), self.E.clone().detach())