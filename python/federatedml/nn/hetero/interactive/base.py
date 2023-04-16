import numpy as np
import federatedml.secureprotol.tensor_util as util
from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


class InteractiveLayerBase(object):

    def __init__(self, params: HeteroNNParam, **kwargs):

        self.params = params
        self.transfer_variable: BaseTransferVariables = None
        self.encrypt_method = params.encrypt_param.method

    def set_flow_id(self, flow_id):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flow_id)

    def set_batch(self, batch_size):
        pass

    def forward(self, x, epoch: int, batch: int, train: bool = True, **kwargs) -> np.ndarray:
        pass

    def backward(self, *args, **kwargs):
        pass

    def guest_backward(self, error, epoch: int, batch_idx: int, **kwargs):
        pass

    def host_backward(self, epoch: int, batch_idx: int, **kwargs):
        pass

    def export_model(self) -> bytes:
        pass

    def restore_model(self, model_bytes: bytes):
        pass

    def set_backward_select_strategy(self):
        pass

    def get_tensor(self, obj, partitions=1):
        return util.get_tensor(obj, partitions, self.encrypt_method)


class InteractiveLayerGuest(InteractiveLayerBase):

    def __init__(self, params: HeteroNNParam, **kwargs):
        super(InteractiveLayerGuest, self).__init__(params, **kwargs)

    def backward(self, error, epoch: int, batch: int, **kwargs):
        pass


class InteractiveLayerHost(InteractiveLayerBase):

    def __init__(self, params: HeteroNNParam, **kwargs):
        super(InteractiveLayerHost, self).__init__(params, **kwargs)

    def backward(self, epoch: int, batch: int, **kwargs):
        pass
