import numpy as np
import tenseal as ts

from federatedml.util import LOGGER
from tenseal.enc_context import Context

class CKKSKeypair(object):
    @staticmethod
    def generate_keypair(poly_modulus_degree=None, coeff_mod_bit_sizes=None, global_scale=2 ** 40):
        """Generate a keypair given security parameters"""
        if poly_modulus_degree is None and coeff_mod_bit_sizes is None:
            poly_modulus_degree = 8192
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = global_scale

        private_key = CKKSPrivateKey(context.copy())

        context.make_context_public()
        public_context = context

        public_key = CKKSPublicKey(public_context)

        return public_key, private_key


class CKKSPublicKey(object):
    def __init__(self, public_context):
        if not isinstance(public_context, Context):
            raise TypeError("public_context should be a tenseal Context object")
        elif not public_context.is_public():
            raise ValueError("public_context is not a public context")

        self.__public_context = public_context

    def __getstate__(self):
        return self.__public_context.serialize(save_public_key=True,
                                               save_secret_key=False,
                                               save_relin_keys=True,
                                               save_galois_keys=False)

    def __setstate__(self, state):
        self.__public_context = ts.context_from(state)

    def encrypt(self, values):
        """Encrypt a scalar or a vector of real-valued number"""
        # Convert to a list of number if values is a scalar
        if self.__is_scalar(values):
            values = [values]

        # Make sure the vector only contains real numbers
        if not self.__only_real_number(values):
            raise ValueError("values should only contain real number")

        ckks_vector = ts.ckks_vector(self.__public_context, values)

        return CKKSEncryptedVector(ckks_vector, self.__public_context)

    def __only_real_number(self, value):
        return np.all(np.isreal(value))

    def __is_scalar(self, obj):
        return isinstance(obj, (np.generic, float, int, str, bytes, complex))


class CKKSPrivateKey(object):
    def __init__(self, context):
        if not (isinstance(context, Context) and context.is_private()):
            raise TypeError("context should be a tenseal Context object")

        self.__context = context

    def decrypt(self, encrypted_vector):
        """Decrypt a CKKSEncryptedNumber"""
        if not isinstance(encrypted_vector, CKKSEncryptedVector):
            raise ValueError("encrypted_vector should be a CKKSEncryptedVector")

        ts_encrypted_vector = encrypted_vector._get_tenseal_encrypted_vector()
        decrypted_vector = ts_encrypted_vector.decrypt(self.__context.secret_key())
        return decrypted_vector[0] if len(decrypted_vector) == 1 else decrypted_vector

    def __getstate__(self):
        return self.__context.serialize(
            save_public_key=False,
            save_secret_key=True,
            save_relin_keys=True,
            save_galois_keys=False)

    def __setstate__(self, state):
        self.__context = ts.context_from(state)
    

class CKKSEncryptedVector(object):
    def __init__(self, encrypted_vector, context, depth=0):
        if not isinstance(encrypted_vector, ts.CKKSVector):
            raise TypeError(f"encrypted_vector should be a tenseal CKKSVector, got {type(encrypted_vector)}")
        elif not isinstance(context, ts.Context):
            raise TypeError(f"context should be a tenseal context, got {type(context)}")

        self.__encrypted_vector = encrypted_vector
        self.__context = context
        self.__depth = depth

    @property
    def depth(self):
        return self.__depth

    def apply_obfuscator(self):
        pass

    def __getstate__(self):
        return self.__encrypted_vector.serialize(), self.__context.serialize(save_public_key=True,
                                                                             save_secret_key=False,
                                                                             save_relin_keys=True,
                                                                             save_galois_keys=False), self.__depth

    def __setstate__(self, state):
        encrypted_vector_bytes, lightweight_context_bytes, self.__depth = state
        self.__context = ts.context_from(lightweight_context_bytes)
        self.__encrypted_vector = ts.ckks_vector_from(self.__context, encrypted_vector_bytes)

    def __add__(self, other):
        if isinstance(other, CKKSEncryptedVector):
            try:
                self.__encrypted_vector += other.__encrypted_vector

            except RuntimeError as e:
                if CKKSEncryptedVector.is_transparent_ciphertext_error(e):
                    self.__encrypted_vector += (other.__encrypted_vector + ts.ckks_vector(other.__context, [0]))
                else:
                    raise e
                
        else:
            self.__encrypted_vector += other
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CKKSEncryptedVector):
            try:
                self.__encrypted_vector = self.__encrypted_vector - other.__encrypted_vector

            except RuntimeError as e:
                if CKKSEncryptedVector.is_transparent_ciphertext_error(e):
                    self.__encrypted_vector -= (other.__encrypted_vector + ts.ckks_vector(other.__context, [0]))
                else:
                    raise e

        else:
            self.__encrypted_vector = self.__encrypted_vector - other
        return self

    def __rsub__(self, other):
        if isinstance(other, CKKSEncryptedVector):
            self.__encrypted_vector = other.__encrypted_vector - self.__encrypted_vector
        else:
            self.__encrypted_vector = other - self.__encrypted_vector
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        try:
            if isinstance(other, CKKSEncryptedVector):
                value = self.__encrypted_vector * other.__encrypted_vector
            else:
                value = self.__encrypted_vector * other
            return CKKSEncryptedVector(value, self.__context, depth=self.__depth + 1)
        except ValueError as e:
            LOGGER.debug(f'Scale out of bound, current depth = {self.__depth+1}')
            raise e
        
    def __truediv__(self, other):
        return self * (1/other)

    def _get_tenseal_encrypted_vector(self):
        """Should only be called by CKKSPrivateKey"""
        return self.__encrypted_vector
    
    @staticmethod
    def is_transparent_ciphertext_error(e: RuntimeError) -> bool:
        '''
        Checks if the specified exception is caused by transparent ciphertext error in tenseal
        '''
        return str(e) == "result ciphertext is transparent"
    
