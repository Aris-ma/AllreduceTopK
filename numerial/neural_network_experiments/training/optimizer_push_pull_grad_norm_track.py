import torch
from torch.optim import Optimizer


class PushPull_grad_norm_track(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, B=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)
        self.B = B.to(next(model_list[0].parameters()).device)

        closure()

        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        defaults = dict(lr=lr)
        super(PushPull_grad_norm_track, self).__init__(
            model_list[0].parameters(), defaults
        )

    def step(self, closure, lr):
        self.lr = lr

        with torch.no_grad():

            prev_params_tensor = torch.stack(self.prev_params)
            v_tensor = torch.stack(self.v_list)

            new_params_tensor = (
                torch.matmul(self.A, prev_params_tensor) - self.lr * v_tensor
            )

            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        for model in self.model_list:
            model.zero_grad()
        loss, grad_norm, avg_grad_norm = closure()

        with torch.no_grad():
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            Bv = torch.matmul(self.B, v_tensor)

            new_v_tensor = Bv + new_grads_tensor - prev_grads_tensor

            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]

        return loss, grad_norm, avg_grad_norm
