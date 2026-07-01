#pragma once
// =============================================================================
// bfv_encrypt.cuh — BFV encrypt / decrypt / ciphertext addition
// =============================================================================

#include "bfv_context.cuh"
#include "bfv_types.cuh"

namespace bfv {

// Encrypt a plaintext under a public key.
// pt must be encoded (pt.encoded populated) before calling.
// ct will be allocated inside this function.
void bfv_encrypt(const BfvContext& ctx, const BfvPublicKey& pk,
                 const BfvPlaintext& pt, BfvCiphertext& ct);

// Decrypt a ciphertext using the secret key.
// pt_out.slots is resized and populated; pt_out.encoded is left untouched.
void bfv_decrypt(const BfvContext& ctx, const BfvSecretKey& sk,
                 const BfvCiphertext& ct, BfvPlaintext& pt_out);

// Ciphertext addition: out = a + b  (component-wise, NTT domain).
// out will be allocated inside this function.
void bfv_add(const BfvContext& ctx,
             const BfvCiphertext& a, const BfvCiphertext& b, BfvCiphertext& out);

} // namespace bfv
