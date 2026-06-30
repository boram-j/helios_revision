// predicates_gpu.cu — COUNT_GPU and SUM_GPU implementations.
//
// COUNT_GPU sums all slots of a BFV ciphertext into one slot via a
// log-depth rotation-and-add tree:
//
//   for step in {1, 2, 4, ..., row_slots/2}:
//       res += rotate_rows(res, step)
//   res += rotate_columns(res)      // swap two rows and add
//
// BFV batch encoding has two rows of (slot_count / 2) slots each.
// Rotation is delegated to a BfvRotationKeyStream so the key management
// strategy can be changed (e.g. via HECache) without touching this file.
//
// Previously the signatures accepted const PhantomGaloisKey&.  They now
// accept BfvRotationKeyStream& (from cache_bridge.h), which wraps exactly
// the same Phantom BFV free functions but through a RotationKeyStream-
// compatible interface.  Callers should construct:
//
//   BfvRotationKeyStream rks(ctx, gk);
//   COUNT_GPU(ctx, ct, slot_count, rks);
//
// SUM_GPU is identical to COUNT_GPU algorithmically; the distinction
// (binary indicator vs integer value) is the caller's responsibility.

#include "nshedb_gpu/predicates/predicates_gpu.cuh"

using namespace phantom;

namespace nshedb_gpu {
namespace predicates {

// ---------------------------------------------------------------------------
// Helper: column-rotation Galois element for BFV batch encoding.
// For poly_modulus_degree N the column swap acts via the automorphism x→x^{2N-1}.
// ---------------------------------------------------------------------------
static uint32_t col_rotation_elt(const PhantomContext &ctx)
{
    // poly_degree_ is a public member of PhantomContext
    return static_cast<uint32_t>(2 * ctx.poly_degree_ - 1);
}

// ---------------------------------------------------------------------------
// COUNT_GPU
// ---------------------------------------------------------------------------

PhantomCiphertext COUNT_GPU(const PhantomContext    &ctx,
                             const PhantomCiphertext &a,
                             size_t                   slot_count,
                             BfvRotationKeyStream    &key_stream)
{
    PhantomCiphertext res = a;

    // Row rotation accumulation: steps 1, 2, 4, …, (slot_count/2 - 1)
    // Each row has slot_count/2 slots; we rotate within one row.
    size_t row_slots = slot_count / 2;
    for (size_t step = 1; step < row_slots; step <<= 1) {
        PhantomCiphertext tmp = res;
        key_stream.rotate_inplace(tmp, static_cast<int>(step));
        add_inplace(ctx, res, tmp);
    }

    // Column swap: add the second row into the first
    PhantomCiphertext rot = res;
    key_stream.apply_galois_inplace_streaming(rot, col_rotation_elt(ctx));
    add_inplace(ctx, res, rot);

    return res;
}

// ---------------------------------------------------------------------------
// SUM_GPU (same algorithm as COUNT_GPU)
// ---------------------------------------------------------------------------

PhantomCiphertext SUM_GPU(const PhantomContext    &ctx,
                           PhantomCiphertext        a,       // by value
                           size_t                   slot_count,
                           BfvRotationKeyStream    &key_stream)
{
    size_t row_slots = slot_count / 2;
    for (size_t step = 1; step < row_slots; step <<= 1) {
        PhantomCiphertext tmp = a;
        key_stream.rotate_inplace(tmp, static_cast<int>(step));
        add_inplace(ctx, a, tmp);
    }

    PhantomCiphertext rot = a;
    key_stream.apply_galois_inplace_streaming(rot, col_rotation_elt(ctx));
    add_inplace(ctx, a, rot);

    return a;
}

} // namespace predicates
} // namespace nshedb_gpu
