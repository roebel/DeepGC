import tensorflow as tf
from debugprint import print_debug


def get_mask_from_lengths(lengths, max_len=None, use_gpu=True):
    lengths = tf.squeeze(lengths, axis=1)
    if max_len is None:
        max_len = tf.reduce_max(lengths)
    ids = tf.range(0, max_len)
    mask = (ids < tf.expand_dims(lengths, axis=1))
    return mask


def test_mask():
    """
    basic test for get_mask_from_lengths function
    """
    lengths = tf.convert_to_tensor([3, 5, 4])
    print_debug(lengths)
    print_debug(tf.math.ceil(lengths / 2))

    data = tf.ones([3, 5, 2])  # [B, T, D]
    print_debug(data)
    # data.fill_(1.)
    mask = get_mask_from_lengths(lengths, data.shape[1])
    print_debug(mask)
    mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, data.shape[2]])   # .float()
    print_debug(mask)
    maskfloat = tf.keras.backend.cast(mask, dtype='float32')
    print_debug(tf.reduce_sum(data * maskfloat)/tf.reduce_sum(maskfloat))
    return mask


if __name__ == '__main__':
    test_mask()
