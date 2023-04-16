from federatedml.util import consts
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.secureprotol.ckks_tensor import CKKSTensor


def get_tensor(obj, partitions=1, type=consts.PAILLIER):
    '''
    Util for building tensors of different encryption schemes
    '''
    if type == consts.PAILLIER:
        return PaillierTensor(obj, partitions)

    elif type == consts.CKKS:
        return CKKSTensor(obj, partitions)

    else:
        raise ValueError(f"tensor type: {type} is not supported")
